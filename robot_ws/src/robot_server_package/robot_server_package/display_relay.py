"""Relay the three Gazebo raw camera topics to D over a dedicated TCP port."""
from __future__ import annotations
import argparse, json, socket, struct, threading, time, uuid
MAGIC=b"ETV1"; HEADER=struct.Struct(">4sI"); MAX_PACKET=8*1024*1024
DEFAULT_HOST="0.0.0.0"; DEFAULT_PORT=47092
CAMERAS={"link":"/link_cam/image","gripper":"/gripper_cam/image","fixed":"/overview_cam/image"}

def pack_packet(metadata,jpeg):
    header=json.dumps(metadata,separators=(",",":")).encode("utf-8")
    body=struct.pack(">I",len(header))+header+jpeg
    if not jpeg or len(body)>MAX_PACKET: raise ValueError("invalid or oversized camera packet")
    return HEADER.pack(MAGIC,len(body))+body

class LatestFrameServer:
    """Keep only each channel's newest packet; ROS callbacks never wait for TCP."""
    def __init__(self,host,port):
        self._lock=threading.Lock(); self._latest={}; self._revision={}; self._client=None; self._stop=threading.Event()
        self._listener=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
        self._listener.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1); self._listener.bind((host,port))
        self._listener.listen(1); self._listener.settimeout(.5)
        threading.Thread(target=self._accept,daemon=True).start(); threading.Thread(target=self._send,daemon=True).start()
    def offer(self,channel,packet):
        with self._lock:
            self._latest[channel]=packet; self._revision[channel]=self._revision.get(channel,0)+1
    def _accept(self):
        while not self._stop.is_set():
            try: client,address=self._listener.accept()
            except socket.timeout: continue
            except OSError: return
            client.settimeout(1.0)
            with self._lock: old,self._client=self._client,client
            if old: old.close()
            print(f"[C 영상] D 접속 {address}",flush=True)
    def _send(self):
        sent_revision={}; active_client=None
        while not self._stop.wait(.01):
            now=time.monotonic()
            with self._lock:
                client=self._client
                packets=[(name,p,self._revision[name]) for name,p in self._latest.items() if self._revision[name]!=sent_revision.get(name,0)]
            if client is None: continue
            if client is not active_client:
                active_client=client; sent_revision.clear()
                with self._lock:
                    packets=[(name,p,self._revision[name]) for name,p in self._latest.items()]
            for name,packet,revision in packets:
                try: client.sendall(packet); sent_revision[name]=revision
                except OSError:
                    with self._lock:
                        if self._client is client: self._client=None
                    client.close(); break
    def close(self):
        self._stop.set(); self._listener.close()
        with self._lock: client,self._client=self._client,None
        if client: client.close()

def main():
    parser=argparse.ArgumentParser(description="Relay Gazebo link, gripper and top-view raw images to D")
    parser.add_argument("--host",default=DEFAULT_HOST); parser.add_argument("--port",type=int,default=DEFAULT_PORT)
    parser.add_argument("--quality",type=int,default=80); args=parser.parse_args()
    if not 1<=args.quality<=100: parser.error("--quality must be between 1 and 100")
    import cv2, rclpy
    from cv_bridge import CvBridge
    from rclpy.executors import ExternalShutdownException
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import Image
    relay=LatestFrameServer(args.host,args.port); bridge=CvBridge(); run_id=str(uuid.uuid4())
    sequences={channel:0 for channel in CAMERAS}; rclpy.init(); node=rclpy.create_node("c_camera_display_relay")
    def subscribe(channel,topic):
        def on_image(message):
            try:
                image=bridge.imgmsg_to_cv2(message,desired_encoding="bgr8")
                ok,encoded=cv2.imencode(".jpg",image,[cv2.IMWRITE_JPEG_QUALITY,args.quality])
                if not ok: return
                sequences[channel]+=1; seq=sequences[channel]
                meta={"channel":channel,"variant":"raw","source_id":f"gz_{channel}","image_seq":seq,
                      "generation":0,"calibration_id":None,"run_id":run_id,"capture_generation":0,
                      "detector_config_id":"raw-camera-v1","job_seq":seq,
                      "stamp_sec":int(message.header.stamp.sec),"stamp_nanosec":int(message.header.stamp.nanosec)}
                relay.offer(channel,pack_packet(meta,encoded.tobytes()))
            except Exception as exc: node.get_logger().warning(f"{channel} frame dropped: {exc}")
        return node.create_subscription(Image,topic,on_image,qos_profile_sensor_data)
    subscriptions=[subscribe(ch,topic) for ch,topic in CAMERAS.items()]
    print(f"[C 영상] {CAMERAS} -> {args.host}:{args.port} 대기 중",flush=True)
    try: rclpy.spin(node)
    except (KeyboardInterrupt,ExternalShutdownException): pass
    finally:
        relay.close(); subscriptions.clear(); node.destroy_node(); rclpy.try_shutdown()
    return 0

if __name__=="__main__": raise SystemExit(main())