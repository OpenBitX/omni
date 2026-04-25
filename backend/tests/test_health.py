"""Smoke test: GET /health returns 200 with expected schema."""
import pytest


@pytest.mark.asyncio
async def test_health_ok(client):
    resp = await client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["ok"] is True
    assert "yolo" in data


@pytest.mark.asyncio
async def test_health_yolo_fields(client):
    resp = await client.get("/health")
    yolo = resp.json()["yolo"]
    assert "model_loaded" in yolo
    assert "device" in yolo
