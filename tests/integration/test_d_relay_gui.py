"""--view relay: MainWindow + RelayViewReceiver against an in-process loopback QTcpServer fixture.

Packets are made by the UNCHANGED C pack_packet. The fixture's link detect/center packets (and the B result on
center) are test data only, not captured from a live C relay. No real C relay, WSL, Gazebo or robot is used; the
command link never opens a socket. Every server/socket/window is closed in tearDown.
"""
import json
import os
import subprocess
import sys
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[2]
GUI_SRC = ROOT / "dashboard-client" / "src"
sys.path.insert(0, str(GUI_SRC))   # canonical D GUI
sys.path.insert(0, str(ROOT / "robot_ws" / "src" / "robot_server_package"))
from PySide6.QtCore import QBuffer, QCoreApplication, QEvent, QIODevice
from PySide6.QtGui import QColor, QImage
from PySide6.QtNetwork import QHostAddress, QTcpServer
from PySide6.QtWidgets import QApplication
from robot_server_package.display_relay import pack_packet
from eyetoarm.gui import motion, relay_view
from eyetoarm.gui.client import StatusLink
from eyetoarm.gui.relay_view import RelayViewReceiver
from eyetoarm.gui.window import MainWindow, NO_VALUE

READY = {"version": "1.1", "type": "STATUS", "robot_ready": True, "not_ready_reasons": [],
         "active_command_id": None, "server_time_ms": 1}


def until(predicate, timeout=5):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        QCoreApplication.processEvents()
        if predicate():
            return True
        time.sleep(0.005)
    return predicate()


def jpeg(color="#3080c0"):
    image = QImage(64, 36, QImage.Format.Format_RGB32)
    image.fill(QColor(color))
    buffer = QBuffer()
    buffer.open(QIODevice.OpenModeFlag.WriteOnly)
    image.save(buffer, "JPG")
    return bytes(buffer.data())


def packet(channel, seq, run="run-A", variant="raw", data=None):
    meta = {"channel": channel, "variant": variant, "source_id": f"gz_{channel}", "image_seq": seq,
            "generation": 0, "calibration_id": None, "run_id": run, "capture_generation": 0,
            "detector_config_id": "raw-camera-v1", "job_seq": seq, "stamp_sec": 1, "stamp_nanosec": 2}
    return pack_packet(meta, data or jpeg())


class RelayWindow(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        motion.set_reduced(True)

    @classmethod
    def tearDownClass(cls):
        motion.set_reduced(False)

    def setUp(self):
        self.server = QTcpServer()
        self.assertTrue(self.server.listen(QHostAddress("127.0.0.1"), 0))
        self.clients = []
        self.server.newConnection.connect(lambda: self.clients.append(self.server.nextPendingConnection()))
        link = StatusLink(log_path=None)
        link.start = Mock()                       # command TCP stays closed in this test
        link._send = Mock(return_value=True)
        link.state, link._conn_id = "connected", 1
        link._handle_line(json.dumps(READY).encode() + b"\n")
        self.link = link
        self.view = RelayViewReceiver("127.0.0.1", self.server.serverPort())
        self.window = MainWindow(link, self.view)
        self.window.show()
        self.now = 1000.0
        self.window.display.clock = lambda: self.now
        self.assertTrue(until(lambda: self.view.state == "connected" and self.clients))

    def tearDown(self):
        self.window.close()
        self.assertFalse(self.view._retry.isActive())
        self.assertFalse(self.view._flush_timer.isActive())
        self.assertEqual(self.view.state, "disconnected")
        for client in self.clients:
            client.abort()
        self.server.close()
        for item in (self.window, self.view, self.link, self.server, *self.clients):
            item.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        self.app.processEvents()

    def send(self, *packets, client=-1):
        socket = self.clients[client]
        socket.write(b"".join(packets))
        socket.flush()

    def settle(self, predicate):
        self.assertTrue(until(predicate))
        self.window._update_status()

    def views(self):
        stage = self.window.stage
        return {"fixed": stage.main, "gripper": stage.gripper, "link": stage.link}

    def mode_of(self, channel):
        return self.views()[channel]._mode

    def test_three_raw_channels_fill_their_own_views_and_status_needs_every_channel(self):
        self.send(packet("fixed", 1))   # first packet is not link: the session (and link source) still starts
        self.settle(lambda: self.view.frame_count == 1)
        self.assertEqual((self.view.run_id, self.window.display.link_source_id), ("run-A", "gz_link"))
        self.assertEqual(self.mode_of("fixed"), "stale")              # first packet may be C's cache: shown, not live
        self.assertIn("확인 대기", self.views()["fixed"]._footer)
        self.assertEqual(self.window.chip_view._level, "fail")
        self.send(packet("fixed", 2))                                 # a newer packet on the same run confirms it
        self.settle(lambda: self.view.frame_count == 2)
        self.assertEqual(self.mode_of("fixed"), "live")
        self.assertNotEqual(self.mode_of("gripper"), "live")
        self.assertEqual(self.window.chip_view._level, "warn")       # TCP connected alone is not "good video"
        self.assertIn("그리퍼", self.window.chip_view._text)
        self.send(packet("gripper", 1), packet("link", 1))
        self.settle(lambda: self.view.frame_count == 4)
        self.send(packet("gripper", 2), packet("link", 2))
        self.settle(lambda: self.view.frame_count == 6)
        self.assertEqual([self.mode_of(c) for c in ("fixed", "gripper", "link")], ["live"] * 3)
        self.assertEqual(self.window.chip_view._level, "ok")
        self.assertTrue(all(value.text() == NO_VALUE for value in self.window.measure_values))   # no results
        self.assertIn("center 를 고르면 B 결과", self.window.stage.popup.values["detail"].toolTip())
        self.assertEqual(self.clients[0].bytesAvailable(), 0)      # D sends nothing until the user picks a link variant

    def test_link_selection_is_display_only_synced_and_never_falls_back_to_raw(self):
        self.send(packet("fixed", 1), packet("gripper", 1), packet("link", 1))
        self.settle(lambda: self.view.frame_count == 3)
        self.send(packet("fixed", 2), packet("gripper", 2), packet("link", 2))
        self.settle(lambda: self.view.frame_count == 6)
        buttons = self.window.mode_buttons
        self.assertEqual(len(buttons), 6)   # setup page + operate page
        detect = [b for b in buttons if b._mode == "detect"]
        detect[-1].click()                  # operate-page selector
        self.window._update_status()
        self.assertEqual({b._mode for b in buttons if b.isChecked()}, {"detect"})
        self.assertEqual(sum(b.isChecked() for b in buttons), 2)        # both pages show the same selection
        self.assertEqual(self.mode_of("link"), "waiting")
        self.assertIn("수신 대기", self.views()["link"]._reason)
        self.assertIn("raw 로 대신 보이지 않음", self.views()["link"]._reason)
        self.assertEqual((self.mode_of("fixed"), self.mode_of("gripper")), ("live", "live"))   # unaffected
        self.assertEqual(self.window.chip_view._level, "warn")
        self.send(packet("link", 1, variant="detect", data=jpeg("#c03030")))   # fixture-only detect packet
        self.settle(lambda: self.view.frame_count == 7)
        self.assertNotEqual(self.mode_of("link"), "live")                       # confirmation is per variant
        self.send(packet("link", 2, variant="detect", data=jpeg("#c03030")))
        self.settle(lambda: self.view.frame_count == 8)
        self.assertEqual(self.mode_of("link"), "live")
        self.assertIn("detect", self.views()["link"]._footer)
        [b for b in buttons if b._mode == "center"][0].click()          # setup-page selector
        self.window._update_status()
        self.assertEqual(self.mode_of("link"), "waiting")
        [b for b in buttons if b._mode == "raw"][0].click()
        self.window._update_status()
        self.assertEqual(self.mode_of("link"), "waiting")              # old raw frames do not count after a switch
        self.send(packet("link", 3))
        self.settle(lambda: self.view.frame_count == 9)
        self.assertNotEqual(self.mode_of("link"), "live")               # first frame may be C's cache
        self.send(packet("link", 4))
        self.settle(lambda: self.view.frame_count == 10)
        self.assertEqual(self.mode_of("link"), "live")
        self.assertIn("raw", self.views()["link"]._footer)
        self.assertEqual(self.requests(3), [self.request(m) for m in ("detect", "center", "raw")])   # one line per click

    def request(self, variant):
        return {"type": "SET_DISPLAY_VARIANT", "channel": "link", "variant": variant}

    def requests(self, count):
        client = self.clients[-1]
        self.assertTrue(until(lambda: client.bytesAvailable() and bytes(client.peek(1 << 16)).count(b"\n") >= count))
        data = bytes(client.readAll())
        self.assertTrue(data.endswith(b"\n"))
        lines = data.split(b"\n")[:-1]
        self.assertTrue(all(b" " not in line for line in lines))        # compact UTF-8 JSON, LF-delimited
        return [json.loads(line) for line in lines]

    def test_link_request_goes_only_to_the_relay_socket_and_shows_sent_or_unsent_without_ack(self):
        center = [b for b in self.window.mode_buttons if b._mode == "center"][-1]
        center.click()
        self.window._update_status()
        self.assertEqual(self.requests(1), [self.request("center")])
        self.link._send.assert_not_called()                              # never the command TCP
        self.assertIn("요청 송신됨 · C 적용 미확인", self.views()["link"]._reason)
        self.send(packet("link", 1))                                     # raw frames are not an ack for center
        self.settle(lambda: self.view.frame_count == 1)
        self.send(packet("link", 2))
        self.settle(lambda: self.view.frame_count == 2)
        self.assertEqual(self.mode_of("link"), "waiting")
        with patch.object(self.view._socket, "write", return_value=-1) as write:
            [b for b in self.window.mode_buttons if b._mode == "center"][0].click()   # same variant again
            self.window._update_status()
        self.assertEqual(write.call_count, 1)
        self.assertIsNone(self.view.requested_variant)                   # earlier success is not left as "sent"
        self.assertEqual(self.view.state, "disconnected")                 # dropped so no partial line can join the next
        self.assertIn("요청 송신 실패", self.view.last_error)
        self.assertNotIn("요청 송신됨", self.views()["link"]._reason)
        self.settle(lambda: self.view.state == "connected" and len(self.clients) == 2)
        self.assertEqual(self.requests(1), [self.request("center")])
        self.assertEqual(self.view.requested_variant, "center")
        self.assertIn("요청 송신됨 · C 적용 미확인", self.views()["link"]._reason)
        with self.assertRaises(ValueError):
            self.view.request_variant("bogus")

    def test_relay_restart_restores_center_and_waits_for_a_fresh_center_frame(self):
        self.window._select_link_mode("center")
        self.assertEqual(self.requests(1), [self.request("center")])
        self.send(packet("link", 1, variant="center"))
        self.settle(lambda: self.view.frame_count == 1)
        self.clients[-1].abort()
        self.settle(lambda: self.view.state == "connected" and len(self.clients) == 2)
        self.assertEqual(self.requests(1), [self.request("center")])
        self.send(packet("link", 1, run="run-B"))
        self.settle(lambda: self.view.frame_count == 2)
        self.assertEqual(self.mode_of("link"), "waiting")
        self.send(packet("link", 1, run="run-B", variant="center"))
        self.settle(lambda: self.view.frame_count == 3)
        self.assertNotEqual(self.mode_of("link"), "live")
        self.send(packet("link", 2, run="run-B", variant="center"))
        self.settle(lambda: self.view.frame_count == 4)
        self.assertEqual(self.mode_of("link"), "live")
        self.assertFalse(until(lambda: self.clients[-1].bytesAvailable() > 0, timeout=0.2))

    def test_first_connection_restores_only_the_last_offline_selection(self):
        # Reset to a disconnected receiver that has never sent a link choice.
        self.view.stop()
        self.window._select_link_mode("center")
        self.assertIsNone(self.view.requested_variant)
        self.view.start()
        self.settle(lambda: self.view.state == "connected" and len(self.clients) == 2)
        self.assertEqual(self.requests(1), [self.request("center")])
        self.view.stop()
        for mode in ("center", "detect", "raw"):
            self.window._select_link_mode(mode)
        self.assertIsNone(self.view.requested_variant)
        self.view.start()
        self.settle(lambda: self.view.state == "connected" and len(self.clients) == 3)
        self.assertEqual(self.requests(1), [self.request("raw")])
        self.assertEqual(self.window.display.mode, "raw")
        self.link._send.assert_not_called()

    def link_color(self):
        frame = self.window.display.frame("link")
        if frame is None:
            return None
        color = frame["image"].pixelColor(4, 4)   # JPEG shifts values slightly: compare the dominant channel
        return max((color.red(), "red"), (color.green(), "green"), (color.blue(), "blue"))[1]

    def click(self, variant):
        [b for b in self.window.mode_buttons if b._mode == variant][-1].click()
        self.window._update_status()

    def feed(self, *packets):
        for item in packets:                       # one write each: the receiver keeps only the newest per turn
            count = self.view.frame_count
            self.send(item)
            self.settle(lambda: self.view.frame_count > count)

    def test_selected_variant_frames_confirm_the_choice_and_other_variants_never_interfere(self):
        meta = dict(fixed={}, gripper={})
        a = dict(generation=3, capture_generation=7, detector_config_id="a-yolo-v9")   # A detect: own numbers
        b = dict(detector_config_id="b-center-overlay-v1")                             # B center: own counter

        def vpacket(seq, variant="raw", color="#3080c0", job=None, **extra):
            value = {"channel": "link", "variant": variant, "source_id": "gz_link", "image_seq": seq,
                     "generation": 0, "calibration_id": None, "run_id": "run-A", "capture_generation": 0,
                     "detector_config_id": "raw-camera-v1", "job_seq": seq if job is None else job,
                     "stamp_sec": 1, "stamp_nanosec": 2}
            value.update(extra)
            return pack_packet(value, jpeg(color))
        self.feed(packet("fixed", 1), packet("gripper", 1), vpacket(1), packet("fixed", 2), packet("gripper", 2), vpacket(2))
        self.assertEqual([self.mode_of(c) for c in ("fixed", "gripper", "link")], ["live"] * 3)

        self.click("center")                                               # quick roundtrip, no center frame
        self.click("raw")
        self.assertEqual([self.mode_of(c) for c in ("fixed", "gripper", "link")], ["live", "live", "waiting"])
        self.assertIsNone(self.window.display.frame("link"))              # cached raw is not shown as live
        self.feed(vpacket(3))
        self.assertNotEqual(self.mode_of("link"), "live")
        self.feed(vpacket(4))
        self.assertEqual(self.mode_of("link"), "live")

        self.click("detect")
        self.assertEqual(self.mode_of("link"), "waiting")
        self.assertIn("요청 송신됨", self.views()["link"]._reason)
        self.feed(vpacket(5))                                              # raw still in flight: ignored
        self.assertEqual(self.mode_of("link"), "waiting")
        self.feed(vpacket(501, "detect", "#c03030", job=40, **a))
        self.assertNotEqual(self.mode_of("link"), "live")
        self.feed(vpacket(502, "detect", "#c03030", job=41, **a), packet("fixed", 3), packet("gripper", 3))
        self.assertEqual([self.mode_of(c) for c in ("fixed", "gripper", "link")], ["live"] * 3)
        self.assertEqual(self.link_color(), "red")
        self.assertEqual(self.window.display.frame("link")["detector_config_id"], "a-yolo-v9")

        self.click("center")                                               # center generation 0 < detect's 3
        self.feed(vpacket(1, "center", "#30c030", **b))
        self.feed(vpacket(503, "detect", "#c03030", job=42, **a))          # late detect (generation 3): ignored
        self.assertNotEqual(self.mode_of("link"), "live")
        self.feed(vpacket(2, "center", "#30c030", **b))
        self.assertEqual(self.mode_of("link"), "live")
        self.assertEqual(self.link_color(), "green")

        self.click("raw")
        self.feed(vpacket(6))                                              # newest raw, but first after the switch
        self.assertNotEqual(self.mode_of("link"), "live")                  # may be C's cache: not live yet
        self.feed(vpacket(7))                                              # raw counter continues independently
        self.assertEqual(self.mode_of("link"), "live")
        self.assertEqual((self.link_color(), self.window.display.frame("link")["image_seq"]), ("blue", 7))
        self.assertEqual((self.view.packet_errors, self.view.decode_errors, self.view.framing_errors), (0, 0, 0))
        self.assertEqual(self.requests(5), [self.request(m) for m in ("center", "raw", "detect", "center", "raw")])

        self.click("detect")                                               # a relay that keeps its selection
        self.assertEqual(self.requests(1), [self.request("detect")])
        self.clients[0].abort()
        self.assertTrue(until(lambda: self.view.state == "connected" and len(self.clients) == 2))
        self.assertEqual(self.window.display.mode, "detect")
        self.assertEqual(self.requests(1), [self.request("detect")])
        self.assertFalse(until(lambda: self.clients[1].bytesAvailable() > 0, timeout=0.3))
        self.feed(vpacket(504, "detect", "#c03030", job=43, **a))         # resent cache after reconnect
        self.assertNotEqual(self.mode_of("link"), "live")
        self.feed(vpacket(505, "detect", "#c03030", job=44, **a))
        self.assertEqual(self.mode_of("link"), "live")

    def test_center_frames_carry_matching_b_results_into_the_measure_card_only(self):
        """C 2026-11-23 display_relay: the selected B result rides on its own center frame's metadata."""
        def center(seq, result=True, **change):
            value = {"channel": "link", "variant": "center", "source_id": "gz_link", "image_seq": seq,
                     "generation": 2, "calibration_id": "calib-1", "run_id": "run-A", "capture_generation": 5,
                     "detector_config_id": "b-det-1", "job_seq": seq, "producer_run_id": "b-run",
                     "capture_stamp_ns": 1_000 + seq, "timestamp_base": "sim_time", "stamp_sec": 1, "stamp_nanosec": 2}
            if result:
                value["result"] = {**{k: value[k] for k in relay_view.RESULT_FRAME_KEYS}, "producer_run_id": "b-run",
                                   "valid": True, "spec": "BOLT_A", "center_px": [320, 240],
                                   "center_w_mm": [12.5, -3.0], "theta_deg": 30.0, "theta_valid": True,
                                   "selected": True, "secret": "dropped", **change}
            return pack_packet(value, jpeg("#30c030"))
        mm, detail = self.window.measure_values[1], self.window.input_values[2]
        self.feed(center(1), center(2))                                     # confirmed, but raw is selected
        self.assertIsNone(self.window.display.result)                       # the window gate drops it
        self.assertIn("center", detail.toolTip())
        self.click("center")
        self.feed(center(3))                                                # first after the switch: C cache?
        self.assertIsNone(self.window.display.result)                       # unconfirmed frame: no result
        self.feed(center(4))
        self.assertIn("12.5", mm.toolTip())
        self.assertIn("BOLT_A", detail.toolTip())
        self.assertNotIn("secret", self.window.display.result)              # only known result keys pass
        drops = self.view.result_drops
        self.feed(center(5, image_seq=99))                                  # result of another frame
        self.assertEqual(self.view.result_drops, drops + 1)
        self.assertEqual((self.mode_of("link"), self.window.display.frame("link")["image_seq"]), ("live", 5))
        self.assertEqual(mm.toolTip(), NO_VALUE)                               # old result no longer matches
        self.feed(center(6, producer_run_id="other"), center(7, result=False))
        self.assertEqual((self.view.result_drops, mm.toolTip()), (drops + 2, NO_VALUE))
        self.feed(center(8))
        self.assertIn("12.5", mm.toolTip())
        self.assertIn("B 결과", self.view.error_summary())
        self.assertEqual((self.view.packet_errors, self.view.decode_errors), (0, 0))

    def test_producer_restart_low_sequence_and_new_generation_are_confirmed_through_the_receiver(self):
        def detect(seq, run, color="#c03030", generation=3, capture=7):
            value = {"channel": "link", "variant": "detect", "source_id": "gz_link", "image_seq": seq,
                     "generation": generation, "calibration_id": None, "run_id": "run-A",
                     "capture_generation": capture, "detector_config_id": "a-yolo-v9", "job_seq": seq,
                     "source_run_id": run, "stamp_sec": 1, "stamp_nanosec": 2}   # C relay A handoff shape
            return pack_packet(value, jpeg(color))
        self.click("detect")
        self.feed(detect(501, "a1"), detect(502, "a1"))
        self.assertEqual(self.mode_of("link"), "live")
        self.feed(detect(1, "a2", "#30c030", capture=0))                   # A restarted: low numbers, new run
        self.assertEqual(self.mode_of("link"), "live")
        frame = self.window.display.frame("link")
        self.assertEqual((frame["image_seq"], frame["source_run_id"], self.link_color()), (1, "a2", "green"))
        drops = self.view.stale_session_drops
        self.send(detect(503, "a1"))                                       # late A1: dropped before decode/display
        self.settle(lambda: self.view.stale_session_drops == drops + 1)
        self.assertEqual(self.window.display.frame("link")["source_run_id"], "a2")
        self.feed(detect(2, "a2", "#30c030", capture=0))
        self.assertEqual((self.mode_of("link"), self.window.display.frame("link")["image_seq"]), ("live", 2))
        self.feed(detect(3, "a2", "#3080c0", generation=4, capture=0))     # same run, new generation
        self.assertEqual((self.mode_of("link"), self.link_color()), ("live", "blue"))
        self.feed(detect(9, "a2", "#c03030", generation=3, capture=0))      # generation regress: refused
        self.assertEqual((self.window.display.frame("link")["generation"], self.link_color()), (4, "blue"))
        self.assertEqual((self.view.packet_errors, self.view.decode_errors, self.view.framing_errors), (0, 0, 0))
        drops, count = self.view.stale_session_drops, self.view.frame_count
        self.send(detect(1, "a3", "#30c030"), detect(504, "a2", "#c03030"))   # one write: late A2 right after A3
        self.settle(lambda: self.view.frame_count == count + 1 and not self.view._flush_timer.isActive())
        frame = self.window.display.frame("link")
        self.assertEqual((frame["source_run_id"], frame["image_seq"], self.link_color()), ("a3", 1, "green"))
        self.assertEqual((self.mode_of("link"), self.view.stale_session_drops), ("live", drops + 1))

    def test_legacy_packets_without_source_run_keep_receiver_order(self):
        self.send(packet("link", 5))
        self.settle(lambda: self.view.frame_count == 1)
        self.assertIsNone(self.window.display.frames[("link", "raw")]["source_run_id"])   # missing key arrives as None
        self.feed(packet("link", 4))                                       # regress: not newer, nothing confirmed
        self.assertIsNone(self.view.frame_age_ms())
        self.assertIsNone(self.view._link_source_run)
        self.feed(packet("link", 6))
        self.assertEqual((self.mode_of("link"), self.window.display.frame("link")["image_seq"]), ("live", 6))

    def test_newest_per_turn_bad_jpeg_and_old_session_cannot_come_back(self):
        self.send(*(packet("link", seq) for seq in range(1, 6)))       # coalesced in one write
        self.settle(lambda: self.view.frame_count >= 1 and not self.view._flush_timer.isActive())
        self.assertEqual(self.view.frame_count, 1)
        self.assertEqual(self.window.display.frame("link")["image_seq"], 5)
        self.send(packet("gripper", 1, data=b"\xff\xd8broken\xff\xd9"))
        self.settle(lambda: self.view.decode_errors == 1)
        self.assertEqual(self.view.state, "connected")
        self.send(packet("fixed", 1, run="run-B"))                       # relay restarted: new session
        self.settle(lambda: self.view.run_id == "run-B")
        self.assertIsNone(self.window.display.frame("link"))             # old run frames are gone
        self.send(packet("link", 9, run="run-A"))                        # an old-run packet must not flip back
        self.settle(lambda: self.view.stale_session_drops == 1)
        self.assertEqual((self.view.run_id, self.window.display.run_id), ("run-B", "run-B"))
        self.assertIsNone(self.window.display.frame("link"))

    def test_corrupt_stream_is_visible_not_just_missing_video(self):
        self.window._update_status()
        self.assertNotIn("수신 오류", self.window.chip_view.toolTip())
        bad_meta = b"{not json"
        body = len(bad_meta).to_bytes(4, "big") + bad_meta + b"\xff\xd8x"
        self.send(b"ETV1" + len(body).to_bytes(4, "big") + body,
                  packet("gripper", 1, data=b"\xff\xd8broken\xff\xd9"))
        self.settle(lambda: self.view.packet_errors == 1 and self.view.decode_errors == 1)
        self.assertIn("수신 오류 있음", self.window.chip_view._text)
        tip = self.window.chip_view.toolTip()
        self.assertIn("메타/패킷 1", tip)
        self.assertIn("JPEG 1", tip)
        self.assertIn("gripper JPEG", tip)                     # the latest error is named, bounded to one line
        self.assertEqual(self.view.state, "connected")

    def test_per_channel_staleness(self):
        self.send(packet("fixed", 1), packet("gripper", 1), packet("link", 1))
        self.settle(lambda: self.view.frame_count == 3)
        self.now = 1002.5
        self.send(packet("gripper", 2))
        self.settle(lambda: self.view.frame_count == 4)
        self.now = 1004.0                                                 # fixed/link 4 s old, gripper 1.5 s
        self.window._update_status()
        self.assertEqual((self.mode_of("fixed"), self.mode_of("gripper"), self.mode_of("link")),
                         ("stale", "live", "stale"))
        self.assertEqual(self.window.chip_view._level, "warn")
        self.assertIn("탑뷰", self.window.chip_view._text)
        self.assertIn("링크", self.window.chip_view._text)
        self.assertNotIn("그리퍼", self.window.chip_view._text)

    def test_disconnect_and_same_run_cached_replay_are_not_revived(self):
        frames = [packet("fixed", 3), packet("gripper", 3), packet("link", 3)]
        self.send(*frames)
        self.settle(lambda: self.view.frame_count == 3)
        self.clients[0].disconnectFromHost()
        self.settle(lambda: self.view.state == "disconnected")
        self.assertTrue(all(self.mode_of(c) == "empty" for c in ("fixed", "gripper", "link")))
        self.assertIn("연결 안 됨", self.views()["fixed"]._reason)
        self.assertNotEqual(self.window.chip_view._level, "ok")
        self.assertTrue(until(lambda: self.view.state == "connected" and len(self.clients) == 2, 6))
        self.send(*frames)                    # C replays its cached newest packets to a new client
        self.settle(lambda: self.view.frame_count == 6)
        self.assertTrue(all(self.window.display.frame(c) is None for c in ("fixed", "gripper", "link")))
        self.assertNotEqual(self.window.chip_view._level, "ok")
        self.send(packet("fixed", 4))
        self.settle(lambda: self.view.frame_count == 7)
        self.assertEqual(self.mode_of("fixed"), "live")

    def test_broken_framing_drops_the_connection_and_retries_with_the_error_kept_visible(self):
        self.send(b"ETV1" + (relay_view.MAX_PACKET + 1).to_bytes(4, "big"))
        self.settle(lambda: self.view.state == "disconnected")
        self.assertTrue(self.view._retry.isActive())
        self.assertEqual(self.view.framing_errors, 1)
        tip = self.window.chip_view.toolTip()
        self.assertIn("경계 1", tip)                                     # visible while disconnected
        self.assertIn(f"127.0.0.1:{self.server.serverPort()}", tip)
        self.assertIn("수신 오류 있음", self.window.chip_view._text)
        self.assertTrue(until(lambda: self.view.state == "connected" and len(self.clients) == 2, 6))
        self.send(b"EAV1" + (10).to_bytes(4, "big"), client=1)          # wrong magic on the new connection
        self.settle(lambda: self.view.framing_errors == 2)
        self.window._update_status()
        self.assertIn("경계 2", self.window.chip_view.toolTip())        # counts persist across reconnects
        self.assertIn("ETV1 아님", self.view.last_error)


    def test_repeats_and_regressions_never_refresh_and_a_new_connection_or_run_starts_unconfirmed(self):
        self.send(packet("fixed", 5))
        self.settle(lambda: self.view.frame_count == 1)
        self.assertIsNone(self.view.frame_age_ms())                     # first packet: cache candidate
        self.send(packet("fixed", 5))
        self.settle(lambda: self.view.frame_count == 2)
        self.send(packet("fixed", 4))
        self.settle(lambda: self.view.frame_count == 3)
        self.assertIsNone(self.view.frame_age_ms())                     # same and older numbers confirm nothing
        self.send(packet("fixed", 6))
        self.settle(lambda: self.view.frame_count == 4)
        self.assertIsNotNone(self.view.frame_age_ms())
        self.assertEqual(self.mode_of("fixed"), "live")
        marked = self.view._last_frame_at
        until(lambda: False, timeout=0.05)
        self.send(packet("fixed", 6))                                   # confirmed channel, repeated number
        self.settle(lambda: self.view.frame_count == 5)
        self.assertEqual(self.view._last_frame_at, marked)
        self.send(packet("gripper", 1, run="run-B"))                    # relay restarted: new run
        self.settle(lambda: self.view.run_id == "run-B")
        self.assertIsNone(self.view.frame_age_ms())
        self.assertEqual(self.window.display.fresh(self.window.display.frame("gripper")), False)
        self.clients[0].disconnectFromHost()
        self.settle(lambda: self.view.state == "disconnected")
        self.assertTrue(until(lambda: self.view.state == "connected" and len(self.clients) == 2, 6))
        self.assertIsNone(self.view.frame_age_ms())                     # nothing carried across the reconnect
        self.send(packet("gripper", 7, run="run-B"))
        self.settle(lambda: self.view.frame_count == 7)
        self.assertIsNone(self.view.frame_age_ms())
        self.assertFalse(self.window.display.fresh(self.window.display.frame("gripper")))


class RelayConnectDeadline(unittest.TestCase):
    """A connect that never completes must not pin the receiver in "connecting". No socket leaves this process:
    connectToHost is replaced, so nothing is sent anywhere (no external address is contacted)."""

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_deadline_drops_retries_and_stop_clears_every_timer(self):
        view = RelayViewReceiver("127.0.0.1", 47092)
        calls = []
        view._socket.connectToHost = lambda *args: calls.append(args)   # never connects, never errors
        view._connect_deadline.setInterval(50)
        view._retry.setInterval(50)
        try:
            view.start()
            self.assertEqual(view.state, "connecting")
            self.assertTrue(view._connect_deadline.isActive())
            self.assertTrue(until(lambda: view.state == "disconnected" and view._retry.isActive(), 2))
            self.assertIn("접속 기한 초과", view.socket_error)
            self.assertTrue(until(lambda: len(calls) >= 2, 2))           # it keeps trying, it is not stuck
            view.stop()
            self.assertEqual(view.state, "disconnected")
            self.assertFalse(view._connect_deadline.isActive())
            self.assertFalse(view._retry.isActive())
            self.assertFalse(view._flush_timer.isActive())
        finally:
            view.stop()
            view.deleteLater()
            QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


CHILD = r"""
import json, sys
from PySide6.QtGui import QFontDatabase
from PySide6.QtWidgets import QAbstractButton, QApplication, QLabel, QScrollArea
from PySide6.QtCore import QPoint, Qt
from eyetoarm.gui import motion, theme
from eyetoarm.gui.demo import DemoLink
from eyetoarm.gui.relay_view import RelayViewReceiver
from eyetoarm.gui.window import MainWindow
app = QApplication([])
for font in ("C:/Windows/Fonts/malgun.ttf", "C:/Windows/Fonts/segoeui.ttf"):
    QFontDatabase.addApplicationFont(font)
theme.apply("dark")                    # the real app style: QSS padding/fonts set the true minimum widths
app.setStyleSheet(theme.QSS)
motion.set_reduced(True)
link = DemoLink()
view = RelayViewReceiver("203.0.113.255", 65535)   # long title; never started, so no socket is opened
view.start = lambda: None
window = MainWindow(link, view, demo=False)
window.show()
clipped, checked, outside = [], 0, []
for size in json.loads(sys.argv[1]):
    window.resize(*size)
    for page in range(3):
        window._show_page(page, animate=False)
        for _ in range(5):
            app.processEvents()
        window._update_status()   # the 250 ms refresh that elides long one-line labels to their current width
        app.processEvents()
        for child in window.pages.widget(page).findChildren(QLabel) + window.pages.widget(page).findChildren(QAbstractButton):
            if not child.isVisible():
                continue
            checked += 1
            text = child.text()
            need_h = child.heightForWidth(child.width()) if child.hasHeightForWidth() else child.minimumSizeHint().height()
            if isinstance(child, QLabel) and not child.wordWrap() and text and child.textFormat() != Qt.TextFormat.RichText:
                m = child.contentsMargins()
                need_w = child.fontMetrics().horizontalAdvance(text) + m.left() + m.right() + 2 * child.margin()
            else:
                need_w = child.minimumSizeHint().width() if isinstance(child, QAbstractButton) else 0
            if child.height() < need_h or child.width() < need_w:
                clipped.append([size, page, type(child).__name__, text[:40], [child.width(), child.height()], [need_w, need_h]])
        # Horizontal geometry guard: the page never scrolls sideways, so content, the video stage and the
        # link selector must end inside the scroll viewport (vertical scrolling is expected and not checked).
        scroll = window.pages.widget(page)
        if isinstance(scroll, QScrollArea):
            viewport = scroll.viewport()
            if scroll.widget().width() > viewport.width():
                outside.append([size, page, "page content", scroll.widget().width(), viewport.width()])
            if page == window.pages.currentIndex():
                for widget in [window.stage, *window.mode_buttons]:
                    if not widget.isVisible() or not viewport.isAncestorOf(widget):
                        continue
                    right = widget.mapTo(viewport, QPoint(widget.width(), 0)).x()
                    if right > viewport.width():
                        outside.append([size, page, type(widget).__name__, right, viewport.width()])
modes = sorted({b._mode for b in window.mode_buttons if b.isVisible() or True})
print(json.dumps({"dpr": window.devicePixelRatio(), "checked": checked, "clipped": clipped, "outside": outside,
                  "size": window.size().toTuple(), "buttons": len(window.mode_buttons), "modes": modes}))
window.close()
for item in (window, link, view):
    item.deleteLater()
app.processEvents()
"""


class RelayCli(unittest.TestCase):
    def run_window(self, *args):
        env = {**os.environ, "QT_QPA_PLATFORM": "offscreen", "PYTHONPATH": str(GUI_SRC), "PYTHONIOENCODING": "utf-8"}
        return subprocess.run([sys.executable, "-B", "-m", "eyetoarm.gui.window", *args], env=env, cwd=str(ROOT),
                              capture_output=True, text=True, encoding="utf-8", timeout=60, stdin=subprocess.DEVNULL)

    def test_relay_is_the_default_and_host_is_validated_before_any_window_or_socket(self):
        result = self.run_window("--help")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("relay(기본)", result.stdout)
        self.assertIn("47092", result.stdout)
        self.assertIn("47091", result.stdout)                    # legacy camera port unchanged
        for args in (("--view-host", "localhost"), ("--view", "camera", "--view-host", "c-pc"),
                     ("--view-port", "70000")):
            with self.subTest(args=args):
                result = self.run_window(*args)                  # parser error: exit 2, nothing started
                self.assertEqual(result.returncode, 2)
                self.assertIn("--view-", result.stderr)

    def test_relay_host_follows_the_command_host_unless_given(self):
        # Separate process; MainWindow is replaced so no window is shown and neither socket is ever started.
        probe = ("import json, sys, tempfile; import eyetoarm.gui.window as w\n"
                 "from PySide6.QtCore import QSettings\n"
                 "tmp = tempfile.TemporaryDirectory(); ini = tmp.name + '/s.ini'\n"   # never the user's registry (saved addresses would change the answer)
                 "w.settings = lambda: QSettings(ini, QSettings.Format.IniFormat)\n"
                 "def fake(link, view, demo=False, store=None):\n"
                 "    print(json.dumps([type(view).__name__, view.host, view.port, link.host])); raise SystemExit(0)\n"
                 "w.MainWindow = fake; w.main(json.loads(sys.argv[1]))\n")
        for args, expected in ((["--host", "10.1.2.3"], ["RelayViewReceiver", "10.1.2.3", 47092, "10.1.2.3"]),
                               (["--host", "10.1.2.3", "--view-host", "10.9.9.9", "--view-port", "47111"],
                                ["RelayViewReceiver", "10.9.9.9", 47111, "10.1.2.3"]),
                               (["--host", "10.1.2.3", "--view", "camera"],
                                ["GzViewReceiver", "127.0.0.1", 47091, "10.1.2.3"])):
            with self.subTest(args=args):
                result = self.run_window_code(probe, json.dumps(args))
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(json.loads(result.stdout.strip().splitlines()[-1]), expected)

    def run_window_code(self, code, *args):
        env = {**os.environ, "QT_QPA_PLATFORM": "offscreen", "PYTHONPATH": str(GUI_SRC), "PYTHONIOENCODING": "utf-8"}
        return subprocess.run([sys.executable, "-B", "-c", code, *args], env=env, cwd=str(ROOT),
                              capture_output=True, text=True, encoding="utf-8", timeout=60, stdin=subprocess.DEVNULL)


class RelayScale(unittest.TestCase):
    def test_relay_controls_fit_at_100_150_and_200_percent_with_the_real_theme(self):
        for scale in (1.0, 1.5, 2.0):
            with self.subTest(scale=scale):
                env = {**os.environ, "QT_QPA_PLATFORM": "offscreen", "QT_SCALE_FACTOR": str(scale),
                       "PYTHONPATH": str(GUI_SRC), "PYTHONIOENCODING": "utf-8"}
                result = subprocess.run([sys.executable, "-B", "-c", CHILD, json.dumps([[1180, 760], [1280, 800]])],
                                        env=env, cwd=str(ROOT), capture_output=True, text=True, encoding="utf-8",
                                        timeout=120, stdin=subprocess.DEVNULL)
                self.assertEqual(result.returncode, 0, result.stderr)
                report = json.loads(result.stdout.strip().splitlines()[-1])
                self.assertEqual(report["dpr"], scale)
                self.assertEqual((report["buttons"], report["modes"]), (6, ["center", "detect", "raw"]))
                self.assertGreater(report["checked"], 100)
                self.assertEqual(report["size"], [1280, 800])        # target size kept, not silently widened
                self.assertEqual(report["clipped"], [])
                self.assertEqual(report["outside"], [])


if __name__ == "__main__":
    unittest.main()
