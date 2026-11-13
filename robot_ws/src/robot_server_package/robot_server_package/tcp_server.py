"""TCP 1.1 listener and request routing."""
from __future__ import annotations
import argparse, asyncio, copy, itertools, json, logging, math, os, time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from .command_log import CommandLog, CommandLogError
from .command_manager import CommandManager, CommandRecord
from .execution_manager import ExecutionManager
from .protocol import (ProtocolError, build_bad_format_status, build_command_status,
    build_get_status_response, build_rejection_status, encode_response, parse_request,
    same_move_content, validate_cancel_request, validate_get_status_request,
    validate_move_request, valid_command_id_from_request)
from .readiness import ReadinessFields, get_readiness_fields
from .robot_backend import UnconfiguredRobotBackend

HOST, PORT, MAX_LINE_BYTES, RECORD_RETENTION, LOG_RETRY_MS = "127.0.0.1", 47090, 4096, 1000, 1000
LOG = logging.getLogger("robot_server_package.tcp_server")

def _project_root() -> Path:
    root = os.environ.get("EYETOARM_ROOT")
    if root: return Path(root).expanduser().resolve()
    for p in (Path.cwd().resolve(), *Path.cwd().resolve().parents, *Path(__file__).resolve().parents):
        if (p / "config/gui/client.json").is_file(): return p
    return Path(__file__).resolve().parents[4]

PROJECT_ROOT = _project_root()
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config/gui/client.json"
DEFAULT_COMMAND_LOG_PATH = PROJECT_ROOT / "logs/command_log.jsonl"

@dataclass
class _ClientSession:
    connection_id: int
    writer: asyncio.StreamWriter
    outgoing: asyncio.Queue[dict[str, Any]] = field(default_factory=asyncio.Queue)
    writer_task: asyncio.Task[None] | None = None
    command_ids: set[str] = field(default_factory=set)
    last_command_status: dict[str, dict[str, Any]] = field(default_factory=dict)
    direct_response_queued: bool = False

class _CancelledBeforeDispatch(Exception): pass

class TcpServer:
    """TCP transport; ROS execution is supplied as an async move_executor."""
    def __init__(self, host: str = HOST, port: int = PORT, max_line_bytes: int = MAX_LINE_BYTES,
        readiness_provider: Callable[[], ReadinessFields] = get_readiness_fields,
        command_log_path: str | Path = DEFAULT_COMMAND_LOG_PATH,
        workspace_mm: dict[str, tuple[float, float]] | None = None,
        move_executor: Callable[..., Any] | None = None,
        record_retention: int = RECORD_RETENTION, log_retry_ms: int = LOG_RETRY_MS) -> None:
        if not isinstance(host, str) or not host or type(port) is not int or not 1 <= port <= 65535:
            raise ValueError("invalid TCP host/port")
        if type(max_line_bytes) is not int or max_line_bytes < 1: raise ValueError("invalid max_line_bytes")
        if type(record_retention) is not int or record_retention < 1: raise ValueError("invalid record_retention")
        if type(log_retry_ms) is not int or log_retry_ms < 1: raise ValueError("invalid log_retry_ms")
        self.host, self.port, self.max_line_bytes = host, port, max_line_bytes
        self.readiness_provider, self.workspace_mm, self.move_executor = readiness_provider, workspace_mm, move_executor
        self.record_retention, self.log_retry_ms = record_retention, log_retry_ms
        self.command_manager = CommandManager(CommandLog(command_log_path), record_retention=record_retention,
                                              notify=self._notify_subscribers)
        self._server: asyncio.AbstractServer | None = None
        self._connections = itertools.count(1)
        self._sessions: dict[int, _ClientSession] = {}
        self._log_retry_task: asyncio.Task[None] | None = None

    def _readiness(self, include_active: bool = True) -> ReadinessFields:
        m = self.command_manager
        if m.recovery_failed: return {"robot_ready": False, "not_ready_reasons": ["RECOVERY_FAILED"], "active_command_id": None}
        base = self.readiness_provider()
        reasons = [x for x in base["not_ready_reasons"] if x not in {"RECOVERING", "COMMAND_ACTIVE"}]
        if m.recovering: reasons.append("RECOVERING")
        if self.workspace_mm is None and "SCENE_NOT_READY" not in reasons: reasons.append("SCENE_NOT_READY")
        if self.move_executor is None and "ROS_DISCONNECTED" not in reasons: reasons.append("ROS_DISCONNECTED")
        if m.log_unavailable and "LOG_UNAVAILABLE" not in reasons: reasons.append("LOG_UNAVAILABLE")
        active = m.active_command_id if include_active and not m.recovering else None
        if active is not None: reasons.append("COMMAND_ACTIVE")
        ready = bool(base["robot_ready"]) and not reasons and active is None
        return {"robot_ready": ready, "not_ready_reasons": reasons if not ready else [], "active_command_id": active}

    def recovery_candidates(self) -> list[dict[str, Any]]:
        m = self.command_manager
        if m.recovery_failed: raise RuntimeError("RECOVERY_FAILED")
        return [{"command_id": r.command_id, "state": r.status.get("state"), "goal_id": r.goal_id,
                 "request": copy.deepcopy(r.request)} for r in m.records.values()
                if m.recovering and r.status.get("state") not in {"REACHED", "FAILED"}]

    def mark_recovery_failed(self, reason: str) -> None:
        m = self.command_manager
        if not m.recovering or not isinstance(reason, str) or not reason.strip(): raise RuntimeError("not recovering or missing reason")
        m.recovery_failed, m.recovering = True, False
        LOG.error("Recovery failed: %s", reason[:256])

    async def complete_recovery(self, *, goals_terminated: bool, robot_stopped: bool) -> None:
        m = self.command_manager
        if not m.recovering or m.recovery_failed: raise RuntimeError("not recovering")
        if not goals_terminated or not robot_stopped: raise ValueError("goals and stopped state must be confirmed")
        for r in list(m.records.values()):
            if r.status.get("state") not in {"REACHED", "FAILED"}:
                await self._set_state(r, "FAILED", stage="RECOVERY", code="SERVER_RESTARTED",
                                      detail="Recovery confirmed after server restart")
        if m.log_unavailable: raise CommandLogError("recovery state is not durably stored")
        m.recovering = False

    async def start(self) -> None:
        if self.command_manager.log_unavailable: self._ensure_log_retry()
        self._server = await asyncio.start_server(self._handle_client, self.host, self.port, limit=self.max_line_bytes + 1)
        LOG.info("TCP server listening at %s", [s.getsockname() for s in self._server.sockets or []])
        async with self._server: await self._server.serve_forever()

    async def close(self) -> None:
        if self._server:
            self._server.close(); await self._server.wait_closed(); self._server = None
        if self._log_retry_task:
            self._log_retry_task.cancel()
            try: await self._log_retry_task
            except asyncio.CancelledError: pass
            self._log_retry_task = None

    def _ensure_log_retry(self) -> None:
        if self._log_retry_task is None or self._log_retry_task.done(): self._log_retry_task = asyncio.create_task(self._retry_log())

    async def _retry_log(self) -> None:
        m = self.command_manager
        while m.log_unavailable and not m.recovery_failed:
            await asyncio.sleep(self.log_retry_ms / 1000)
            try: m.command_log.probe()
            except CommandLogError as e: LOG.warning("Log probe failed: %s", e); continue
            if not m.pending_terminal_statuses: m.log_unavailable = False

    async def _handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        cid = next(self._connections); session = _ClientSession(cid, writer); self._sessions[cid] = session
        session.writer_task = asyncio.create_task(self._writer_loop(session))
        buf = bytearray(); discard = False; signature: list[str | None] = [None]
        try:
            while chunk := await reader.read(4096):
                pos = 0
                while pos < len(chunk):
                    if discard:
                        end = chunk.find(b"\n", pos)
                        if end < 0: break
                        await self._bad_format(session); pos, discard = end + 1, False; continue
                    end = chunk.find(b"\n", pos)
                    if end < 0:
                        buf.extend(chunk[pos:])
                        if len(buf) > self.max_line_bytes: buf.clear(); discard = True
                        break
                    part = chunk[pos:end]
                    if len(buf) + len(part) > self.max_line_bytes:
                        buf.clear(); await self._bad_format(session)
                    else:
                        line = bytes(buf) + part; buf.clear()
                        await self._handle_line(line, session, signature)
                    pos = end + 1
        except (ConnectionError, asyncio.CancelledError): raise
        except Exception: LOG.exception("Client handler failed: %s", cid)
        finally:
            if session.writer_task:
                session.writer_task.cancel()
                try: await session.writer_task
                except asyncio.CancelledError: pass
            writer.close()
            try: await writer.wait_closed()
            except ConnectionError: pass
            self._sessions.pop(cid, None)
            for key in session.command_ids:
                r = self.command_manager.records.get(key)
                if r: r.subscribers.discard(cid)

    async def _handle_line(self, line: bytes, session: _ClientSession, previous: list[str | None]) -> None:
        session.direct_response_queued = False; request: dict[str, Any] = {}
        try:
            request = parse_request(line); kind = request.get("type"); ready = self._readiness()
            if kind == "GET_STATUS":
                key = validate_get_status_request(request)
                if key is None: response = build_get_status_response(request, ready)
                else:
                    rec = self.command_manager.find_record(key)
                    response = await self._snapshot(session, rec) if rec else build_rejection_status("UNKNOWN_COMMAND", self._readiness(), key)
            elif kind == "MOVE":
                try: valid = validate_move_request(request)
                except ProtocolError as e: response = build_rejection_status(e.code, ready, valid_command_id_from_request(request), detail=str(e)[:256])
                else: response = await self._move(valid, session, ready)
            elif kind == "CANCEL": response = await self._cancel(validate_cancel_request(request), session, ready)
            else: raise ProtocolError("Unsupported request type")
        except ProtocolError as e:
            response = build_rejection_status(e.code, self._readiness(), valid_command_id_from_request(request), detail=str(e)[:256])
        sig = json.dumps({k:v for k,v in response.items() if k != "server_time_ms"}, sort_keys=True)
        if sig != previous[0]: LOG.info("STATUS %s", sig); previous[0] = sig
        if not session.direct_response_queued: self._queue(session, response)
        if response.get("state") == "ACK" and request.get("type") == "MOVE":
            r = self.command_manager.records.get(response["command_id"])
            if r and r.task is None: r.task = asyncio.create_task(self._run_move(r))

    async def _snapshot(self, session: _ClientSession, r: CommandRecord) -> dict[str, Any]:
        async with r.lock:
            if r.status.get("state") not in {"REACHED", "FAILED"}:
                r.subscribers.add(session.connection_id); session.command_ids.add(r.command_id)
            result = self._now(r); self._queue(session, result); session.direct_response_queued = True
            return result

    @staticmethod
    def _now(r: CommandRecord) -> dict[str, Any]:
        result = dict(r.status); result["server_time_ms"] = time.monotonic_ns() // 1_000_000; return result

    def _workspace_error(self, req: dict[str, Any]) -> str | None:
        if self.workspace_mm is None: return "NOT_CONFIGURED"
        axes = ("x", "y") if req["mode"] == "PLANAR" else ("x", "y", "z")
        for axis, value in zip(axes, req["position_mm"]):
            b = self.workspace_mm.get(axis)
            if not isinstance(b, (list,tuple)) or len(b)!=2 or any(type(x) not in (int,float) or not math.isfinite(x) for x in b) or b[0]>b[1]: return "NOT_CONFIGURED"
            if not b[0] <= float(value) <= b[1]: return "OUT_OF_WORKSPACE"
        return None

    async def _move(self, req: dict[str, Any], session: _ClientSession, ready: ReadinessFields) -> dict[str, Any]:
        key=req["command_id"]; m=self.command_manager; w=self._workspace_error(req)
        if w == "OUT_OF_WORKSPACE": return build_rejection_status(w, ready, key)
        if w == "NOT_CONFIGURED": return build_rejection_status("NOT_READY", self._readiness(), key, detail="workspace_mm not configured")
        if m.recovery_failed: return build_rejection_status("NOT_READY", self._readiness(), key)
        async with m.commands_lock:
            old=m.find_record(key)
            if old:
                if same_move_content(old.request, req): return await self._snapshot(session, old)
                return build_rejection_status("ID_CONFLICT", self._readiness(), key)
            current=self._readiness()
            if m.active_command_id is not None: return build_rejection_status("BUSY", current, key)
            if not current["robot_ready"]: return build_rejection_status("NOT_READY", current, key)
            if self.move_executor is None: return build_rejection_status("NOT_READY", {"robot_ready":False,"not_ready_reasons":["ROS_DISCONNECTED"],"active_command_id":None}, key)
            if m.recovering: return build_rejection_status("NOT_READY", current, key)
            ack=build_command_status(key,"ACK",{"robot_ready":False,"not_ready_reasons":["COMMAND_ACTIVE"],"active_command_id":key})
            try: r=m.register(req,ack)
            except CommandLogError as e:
                self._ensure_log_retry(); return build_rejection_status("NOT_READY",self._readiness(),key,detail=f"LOG_UNAVAILABLE: {e}"[:256])
            async with r.lock:
                r.subscribers.add(session.connection_id); session.command_ids.add(key)
                self._queue(session,ack); session.direct_response_queued=True
            return ack

    async def _cancel(self, key: str, session: _ClientSession, ready: ReadinessFields) -> dict[str, Any]:
        r=self.command_manager.find_record(key)
        if r is None: return build_rejection_status("UNKNOWN_COMMAND",ready,key)
        if self.command_manager.recovering: return await self._snapshot(session,r)
        async with r.dispatch_lock:
            async with r.lock:
                pending=key in self.command_manager.pending_terminal_statuses
                if r.status.get("state") not in {"REACHED","FAILED"}:
                    r.subscribers.add(session.connection_id); session.command_ids.add(key)
                    if not pending: r.cancel_event.set()
                response=self._now(r); self._queue(session,response); session.direct_response_queued=True; return response

    async def _run_move(self, r: CommandRecord) -> None:
        try:
            await self._set_state(r,"PLANNING")
            if r.cancel_event.is_set(): await self._set_state(r,"FAILED",stage="PLANNING",code="CANCELLED"); return
            async def report(state: str, *, stage=None, code=None, detail=None, reach_error=None, goal_id=None, dispatch=None):
                if state != "MOVING":
                    if dispatch is not None or goal_id is not None: raise ValueError("goal_id/dispatch only valid for MOVING")
                    return await self._set_state(r,state,stage=stage,code=code,detail=detail,reach_error=reach_error)
                if not goal_id or dispatch is None: raise ValueError("MOVING requires goal_id and dispatch")
                async with r.dispatch_lock:
                    if r.cancel_event.is_set():
                        await self._set_state(r,"FAILED",stage="PLANNING",code="CANCELLED"); raise _CancelledBeforeDispatch()
                    await self._set_state(r,"MOVING",goal_id=goal_id,stage=stage,detail=detail)
                    return await dispatch()
            await self.move_executor(copy.deepcopy(r.request),report,r.cancel_event)
            if r.status.get("state") not in {"REACHED","FAILED"}:
                moving=r.status.get("state")=="MOVING"
                if not moving and r.cancel_event.is_set(): await report("FAILED",stage="PLANNING",code="CANCELLED")
                else: await report("FAILED",stage="EXECUTION" if moving else "PLANNING",code="EXEC_FAILED" if moving else "PLAN_FAILED",detail="executor returned without terminal state")
        except _CancelledBeforeDispatch: return
        except asyncio.CancelledError: raise
        except CommandLogError as e:
            if r.status.get("state") not in {"REACHED","FAILED"}:
                moving=r.status.get("state")=="MOVING"
                await self._set_state(r,"FAILED",stage="EXECUTION" if moving else "PLANNING",code="LOG_WRITE_FAILED",detail=str(e)[:256])
        except Exception as e:
            LOG.exception("MOVE executor failed: %s",r.command_id)
            if r.status.get("state") not in {"REACHED","FAILED"}:
                moving=r.status.get("state")=="MOVING"
                await self._set_state(r,"FAILED",stage="EXECUTION" if moving else "PLANNING",code="EXEC_FAILED" if moving else "PLAN_FAILED",detail=str(e)[:256])

    async def _set_state(self,r: CommandRecord,state: str,*,stage=None,code=None,detail=None,reach_error=None,goal_id=None):
        if state in {"MOVING","PLANNING"}: ready={"robot_ready":False,"not_ready_reasons":["COMMAND_ACTIVE"],"active_command_id":r.command_id}
        else: ready=self._readiness(include_active=False)
        await self.command_manager.transition(r,state,ready,stage=stage,code=code,detail=detail,reach_error=reach_error,goal_id=goal_id,retry_ms=self.log_retry_ms)

    async def _notify_subscribers(self,r: CommandRecord,status: dict[str,Any]):
        terminal=status.get("state") in {"REACHED","FAILED"}
        for cid in tuple(r.subscribers):
            s=self._sessions.get(cid)
            if s is None: r.subscribers.discard(cid); continue
            self._queue(s,status)
            if terminal: r.subscribers.discard(cid); s.command_ids.discard(r.command_id)

    async def _bad_format(self,s:_ClientSession): self._queue(s,build_bad_format_status(self._readiness()))

    def _queue(self,s:_ClientSession,response:dict[str,Any]):
        out=dict(response); key=out.get("command_id"); state=out.get("state")
        if isinstance(key,str) and isinstance(state,str):
            rank={"ACK":0,"PLANNING":1,"MOVING":2,"REACHED":3,"FAILED":3}; old=s.last_command_status.get(key); oldstate=old.get("state") if old else None
            if oldstate in rank and (rank[oldstate]>rank.get(state,-1) or (rank[oldstate]==3 and state!=oldstate)):
                out=dict(old); out["server_time_ms"]=time.monotonic_ns()//1_000_000
            s.last_command_status[key]=dict(out)
            if len(s.last_command_status)>self.record_retention:
                for old_id, old_status in tuple(s.last_command_status.items()):
                    if old_status.get("state") in {"REACHED","FAILED"} and old_id!=key:
                        s.last_command_status.pop(old_id,None)
                        if len(s.last_command_status)<=self.record_retention: break
        s.outgoing.put_nowait(out)

    async def _writer_loop(self,s:_ClientSession):
        try:
            while True:
                item=await s.outgoing.get()
                try: s.writer.write(encode_response(item)); await s.writer.drain()
                finally: s.outgoing.task_done()
        except asyncio.CancelledError: raise
        except (ConnectionError,OSError): LOG.info("TCP writer disconnected: %s",s.connection_id)
        finally:
            if not s.writer.is_closing(): s.writer.close()


def load_tcp_settings(config_path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str,Any]:
    path=Path(config_path).expanduser().resolve()
    try: root=json.loads(path.read_text(encoding="utf-8")); tcp=root["tcp"]
    except (OSError,UnicodeDecodeError,json.JSONDecodeError,KeyError,TypeError) as e: raise ValueError(f"cannot read TCP config {path}: {e}") from e
    if not isinstance(tcp,dict): raise ValueError("tcp settings must be an object")
    host=tcp.get("host",HOST); port=tcp.get("port",PORT); maximum=tcp.get("max_line_bytes",MAX_LINE_BYTES)
    retention=tcp.get("record_retention",RECORD_RETENTION); retry=tcp.get("log_retry_ms",LOG_RETRY_MS)
    if not isinstance(host,str) or not host or type(port) is not int or not 1<=port<=65535: raise ValueError("invalid TCP host/port")
    if any(type(x) is not int or x<1 for x in (maximum,retention,retry)): raise ValueError("TCP integer settings must be positive")
    if tcp.get("command_log_retention","project_lifetime")!="project_lifetime": raise ValueError("command log must be retained for project lifetime")
    lp=Path(tcp.get("command_log_path","logs/command_log.jsonl")).expanduser(); lp=lp if lp.is_absolute() else PROJECT_ROOT/lp
    workspace=tcp.get("workspace_mm")
    if workspace is not None:
        if not isinstance(workspace,dict): raise ValueError("workspace_mm must be an object")
        for axis in ("x","y","z"):
            b=workspace.get(axis)
            if not isinstance(b,list) or len(b)!=2 or any(type(v) not in (int,float) or not math.isfinite(v) for v in b) or b[0]>b[1]: raise ValueError(f"invalid workspace_mm.{axis}")
    return {"host":host,"port":port,"max_line_bytes":maximum,"record_retention":retention,"log_retry_ms":retry,"command_log_path":lp.resolve(),"workspace_mm":workspace}

async def serve(host=None,port=None,*,config_path=DEFAULT_CONFIG_PATH,readiness_provider=get_readiness_fields,workspace_mm=None,move_executor=None,execution_manager=None):
    cfg=load_tcp_settings(config_path)
    if host is not None: cfg["host"]=host
    if port is not None: cfg["port"]=port
    if workspace_mm is not None: cfg["workspace_mm"]=workspace_mm
    if execution_manager is not None:
        readiness_provider=execution_manager.readiness_provider
        move_executor=execution_manager.execute_move
    server=TcpServer(**cfg,readiness_provider=readiness_provider,move_executor=move_executor)
    try: await server.start()
    finally:
        try:
            await server.close()
        finally:
            if execution_manager is not None:
                await execution_manager.close()

def main():
    parser=argparse.ArgumentParser(description="EyetoArm TCP 1.1 server"); parser.add_argument("--config",default=str(DEFAULT_CONFIG_PATH)); parser.add_argument("--initialize-command-log",action="store_true"); args=parser.parse_args()
    logging.basicConfig(level=logging.INFO,format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        cfg=load_tcp_settings(args.config)
        if args.initialize_command_log: CommandLog.initialize(cfg["command_log_path"]); return
        backend=UnconfiguredRobotBackend()
        manager=ExecutionManager(backend)
        asyncio.run(serve(config_path=args.config,execution_manager=manager))
    except KeyboardInterrupt: LOG.info("Server shutdown requested")
    except (ValueError,CommandLogError) as e: LOG.error("TCP configuration failed: %s",e); raise SystemExit(2) from e

if __name__=="__main__": main()
