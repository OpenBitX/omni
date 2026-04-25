"""Smoke test: /ws/yolo WebSocket protocol handshake and message schema."""
import pytest


def test_yolo_ws_ready(sync_client, tiny_jpeg):
    """Connect to /ws/yolo, expect {type:'ready'}, send a frame, expect {type:'detections'}."""
    with sync_client.websocket_connect("/ws/yolo") as ws:
        ready = ws.receive_json()
        assert ready.get("type") == "ready", f"Expected 'ready', got: {ready}"
        assert "model" in ready
        assert "device" in ready
        assert "imgsz" in ready

        ws.send_json({"type": "opts", "seq": 1, "conf": 0.35, "iou": 0.45, "maxDet": 5})
        ws.send_bytes(tiny_jpeg)

        dets = ws.receive_json()
        assert dets.get("type") == "detections", f"Expected 'detections', got: {dets}"
        assert "detections" in dets
        assert isinstance(dets["detections"], list)
        assert "inferMs" in dets
        assert "w" in dets
        assert "h" in dets
