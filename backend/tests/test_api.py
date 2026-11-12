from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from . import conftest as fx
from .conftest import upload
from .test_inspection import _Exploit


def test_health(client):
    body = client.get("/api/v1/health").json()
    assert body["status"] == "ok" and body["models"] == 0 and body["tts_ready"] is True


def test_import_rvc_end_to_end(client, settings):
    res = upload(client, fx.rvc_zip(), "MyVoice.zip")
    assert res.status_code == 201, res.text
    model = res.json()
    assert model["name"] == "MyVoice" and model["version"] == 1
    assert model["architecture"] == "rvc" and model["kind"] == "voice_conversion"
    assert model["standalone_tts"] is False
    from app.engines.rvc.engine import RVCEngine

    assert model["inference"]["ready"] is RVCEngine.is_available()[0]
    assert model["license_status"] == "unknown"
    assert any("License unknown" in w for w in model["warnings"])
    assert (Path(settings.models_dir) / model["id"] / "MyVoice" / "MyVoice.pth").exists()
    assert not any(settings.staging_dir.iterdir())

    assert [m["id"] for m in client.get("/api/v1/models").json()] == [model["id"]]
    assert client.get(f"/api/v1/models/{model['id']}").json()["report"]["checkpoints"][0]["verdict"] == "safe"


def test_duplicate_and_versioning(client):
    assert upload(client, fx.rvc_zip(), "Voice.zip").status_code == 201
    assert upload(client, fx.rvc_zip(), "Voice.zip").status_code == 409
    other = fx.make_zip({"Voice.pth": fx.torch_zip_bytes({**fx.RVC, "info": "500epoch"})})
    assert upload(client, other, "Voice.zip").json()["version"] == 2


def test_malicious_checkpoint_is_rejected_and_not_installed(client, settings, tmp_path):
    marker = tmp_path / "pwned"
    evil = fx.make_zip({"evil.pth": fx.torch_zip_bytes({"weight": fx.RVC["weight"], "x": _Exploit(marker)})})
    res = upload(client, evil)
    assert res.status_code == 422 and "execute code" in res.json()["detail"]
    assert not marker.exists()
    assert client.get("/api/v1/models").json() == []
    assert not any(settings.models_dir.iterdir())
    assert client.get("/api/v1/system/audit").json()[0]["action"] == "model.import.rejected"


def test_bad_uploads(client):
    assert upload(client, b"x", "model.rar").status_code == 415
    assert upload(client, b"not a zip", "model.zip").status_code == 422
    assert upload(client, fx.make_zip({"../x.pth": b"x"})).status_code == 422


def test_activation_rules_metadata_and_delete(client, settings):
    piper = upload(client, fx.piper_zip(), "ml_IN-test.zip").json()
    assert piper["languages"] == ["ml"] and piper["license"] == "Apache-2.0"
    res = client.patch(f"/api/v1/models/{piper['id']}", json={"active": True, "tags": ["malayalam", " teacher "], "source_url": "https://example.org/voice"})
    assert res.status_code == 200 and res.json()["active"] is True and res.json()["tags"] == ["malayalam", "teacher"]

    half = upload(client, fx.make_zip({"s.pth": fx.torch_zip_bytes(fx.SOVITS)}), "half.zip").json()
    assert client.patch(f"/api/v1/models/{half['id']}", json={"active": True}).status_code == 409

    assert client.delete(f"/api/v1/models/{piper['id']}").status_code == 204
    assert not (settings.models_dir / piper["id"]).exists()
    assert client.get(f"/api/v1/models/{piper['id']}").status_code == 404


def test_api_key_guards_writes(settings, monkeypatch):
    from app.config import get_settings
    from app.main import create_app

    monkeypatch.setenv("VOIX_API_KEY", "secret")
    get_settings.cache_clear()
    client = TestClient(create_app())
    assert client.get("/api/v1/models").status_code == 200
    assert upload(client, fx.rvc_zip()).status_code == 401
    assert upload(client, fx.rvc_zip(), headers={"X-API-Key": "secret"}).status_code == 201


def test_characters_crud(client):
    res = client.post("/api/v1/characters", json={"name": "Friendly Teacher", "default_speed": 0.95})
    assert res.status_code == 201 and res.json()["id"] == "friendly_teacher"
    assert client.post("/api/v1/characters", json={"name": "Friendly Teacher"}).status_code == 409
    assert client.post("/api/v1/characters", json={"name": "Robot", "voice_ml": "nope"}).status_code == 422
    assert client.put("/api/v1/characters/friendly_teacher", json={"name": "Friendly Teacher", "description": "Warm"}).json()["description"] == "Warm"
    assert client.delete("/api/v1/characters/friendly_teacher").status_code == 204
    assert client.get("/api/v1/characters").json() == []
