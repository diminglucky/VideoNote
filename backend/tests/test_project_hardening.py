import io
import json
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import HTTPException, UploadFile

from app.routers import note as note_router
from app.services import provider as provider_service


def _response_payload(response):
    return json.loads(response.body)


def test_provider_list_masks_api_keys(monkeypatch):
    row = SimpleNamespace(
        id="provider-1",
        name="demo",
        logo="custom",
        type="custom",
        api_key="sk-secret-1234",
        base_url="https://api.example.com/v1",
        enabled=1,
        created_at=None,
    )
    monkeypatch.setattr(provider_service, "get_all_providers", lambda: [row])

    payload = provider_service.ProviderService.get_all_providers_safe()

    assert payload[0]["api_key"] == "sk-s******1234"
    assert "secret" not in payload[0]["api_key"]


def test_upload_rejects_path_traversal_and_streams_inside_upload_dir(
    monkeypatch,
    tmp_path,
):
    upload_dir = tmp_path / "uploads"
    monkeypatch.setattr(note_router, "UPLOAD_DIR", str(upload_dir))
    monkeypatch.setattr(note_router, "UPLOAD_MAX_BYTES", 1024)
    upload = UploadFile(
        file=io.BytesIO(b"video-data"),
        filename="../escape.mp4",
    )

    response = __import__("asyncio").run(note_router.upload(upload))
    payload = _response_payload(response)

    assert payload["code"] == 0
    returned_url = payload["data"]["url"]
    stored_name = returned_url.rsplit("/", 1)[-1]
    assert stored_name.endswith(".mp4")
    assert "escape" in stored_name
    assert (upload_dir / stored_name).read_bytes() == b"video-data"
    assert not (tmp_path / "escape.mp4").exists()


def test_upload_enforces_size_limit(monkeypatch, tmp_path):
    monkeypatch.setattr(note_router, "UPLOAD_DIR", str(tmp_path / "uploads"))
    monkeypatch.setattr(note_router, "UPLOAD_MAX_BYTES", 4)
    upload = UploadFile(
        file=io.BytesIO(b"12345"),
        filename="large.mp4",
    )

    response = __import__("asyncio").run(note_router.upload(upload))
    payload = _response_payload(response)

    assert payload["code"] == 413
    assert "超过" in payload["msg"]


def test_upload_rejects_non_media_extension(monkeypatch, tmp_path):
    upload_dir = tmp_path / "uploads"
    monkeypatch.setattr(note_router, "UPLOAD_DIR", str(upload_dir))
    upload = UploadFile(
        file=io.BytesIO(b"<script>alert(1)</script>"),
        filename="payload.html",
    )

    response = __import__("asyncio").run(note_router.upload(upload))
    payload = _response_payload(response)

    assert payload["code"] == 400
    assert "不支持" in payload["msg"]
    assert not upload_dir.exists() or not any(upload_dir.iterdir())


def test_local_image_proxy_rejects_path_escape(monkeypatch, tmp_path):
    static_dir = tmp_path / "static"
    uploads_dir = tmp_path / "uploads"
    allowed = static_dir / "screenshots" / "ok.png"
    allowed.parent.mkdir(parents=True)
    allowed.write_bytes(b"image")
    uploads_dir.mkdir()
    monkeypatch.setattr(note_router, "STATIC_DIR", static_dir)
    monkeypatch.setattr(note_router, "UPLOADS_DIR", uploads_dir)

    assert note_router._resolve_local_image_proxy_path(
        "/static/screenshots/ok.png"
    ) == allowed.resolve()

    with pytest.raises(HTTPException):
        note_router._resolve_local_image_proxy_path("/static/../secret.txt")


def test_local_image_proxy_respects_configured_static_prefix(monkeypatch, tmp_path):
    static_dir = tmp_path / "static"
    allowed = static_dir / "ok.png"
    allowed.parent.mkdir(parents=True)
    allowed.write_bytes(b"image")
    monkeypatch.setattr(note_router, "STATIC_DIR", static_dir)
    monkeypatch.setattr(note_router, "STATIC_URL_PREFIX", "/media")

    assert note_router._resolve_local_image_proxy_path("/media/ok.png") == allowed.resolve()


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1/private.png",
        "http://localhost/private.png",
        "http://169.254.169.254/latest/meta-data",
        "http://10.0.0.1/private.png",
        "file:///etc/passwd",
    ],
)
def test_remote_image_proxy_rejects_non_public_targets(url):
    with pytest.raises(HTTPException):
        note_router._validate_remote_image_url(url)


def test_remote_image_proxy_allows_public_http_url(monkeypatch):
    monkeypatch.setattr(
        note_router.socket,
        "getaddrinfo",
        lambda *_args, **_kwargs: [
            (note_router.socket.AF_INET, note_router.socket.SOCK_STREAM, 6, "", ("93.184.216.34", 0))
        ],
    )
    note_router._validate_remote_image_url("https://example.com/image.png")


@pytest.mark.parametrize(
    ("content_type", "expected"),
    [
        ("image/png", True),
        ("image/jpeg; charset=binary", True),
        ("image/webp", True),
        ("image/svg+xml", False),
        ("text/html", False),
        ("", False),
    ],
)
def test_image_proxy_only_accepts_raster_content_types(content_type, expected):
    assert note_router._is_allowed_image_content_type(content_type) is expected


def test_delete_task_removes_known_artifacts_only(tmp_path):
    task_id = uuid.uuid4()
    known_paths = [
        tmp_path / f"{task_id}.json",
        tmp_path / f"{task_id}.status.json",
        tmp_path / f"{task_id}_markdown.md",
        tmp_path / f"{task_id}_transcript.json",
        tmp_path / f"{task_id}_audio.json",
        tmp_path / f"{task_id}.agent-trace.jsonl",
        tmp_path / f"{task_id}.gpt.checkpoint.json",
    ]
    for path in known_paths:
        path.write_text("data", encoding="utf-8")
    untouched = tmp_path / f"{task_id}.keep"
    untouched.write_text("keep", encoding="utf-8")

    removed = note_router._delete_task_artifacts(task_id, output_dir=tmp_path)

    assert removed == len(known_paths)
    assert not any(path.exists() for path in known_paths)
    assert untouched.exists()


def test_delete_task_calls_storage_and_index_cleanup(monkeypatch, tmp_path):
    task_id = uuid.uuid4()
    calls = []

    class FakeVectorStore:
        def delete_index(self, value):
            calls.append(("index", value))

    monkeypatch.setattr(
        note_router,
        "_delete_task_artifacts",
        lambda value, output_dir=None: calls.append(("artifacts", str(value))) or 3,
    )
    monkeypatch.setattr(
        note_router.NoteGenerator,
        "delete_note",
        staticmethod(lambda video_id, platform: calls.append(("db", video_id, platform)) or 1),
    )
    monkeypatch.setattr(note_router, "VectorStoreManager", FakeVectorStore)

    response = note_router.delete_task(
        note_router.RecordRequest(
            task_id=str(task_id),
            video_id="video-1",
            platform="bilibili",
        )
    )
    payload = _response_payload(response)

    assert payload["code"] == 0
    assert calls == [
        ("artifacts", str(task_id)),
        ("index", str(task_id)),
        ("db", "video-1", "bilibili"),
    ]


def test_complete_image_serves_frontend_without_missing_port():
    root = Path(__file__).resolve().parents[2]
    dockerfile = (root / "Dockerfile.complete").read_text(encoding="utf-8")
    aio_nginx = (root / "nginx" / "aio.conf").read_text(encoding="utf-8")

    assert "nginx/aio.conf" in dockerfile
    assert "proxy_pass http://127.0.0.1:8080" not in dockerfile
    assert "try_files $uri /index.html" in aio_nginx
    assert "proxy_pass http://127.0.0.1:8483" in aio_nginx
    assert "BILINOTE_LLM_AGENT_ENABLED" in dockerfile
    assert "%(ENV_BILINOTE_LLM_AGENT_ENABLED)s" in dockerfile
    assert "SCREENSHOT_REVIEW_MODE" in dockerfile
    assert "%(ENV_SCREENSHOT_REVIEW_MODE)s" in dockerfile
    assert 'VECTOR_DB_DIR="data/vector_db"' in dockerfile

    env_example = (root / ".env.example").read_text(encoding="utf-8")
    assert "VECTOR_DB_DIR=data/vector_db" in env_example


def test_compose_binds_to_loopback_by_default():
    root = Path(__file__).resolve().parents[2]
    compose = (root / "docker-compose.yml").read_text(encoding="utf-8")

    assert "${APP_BIND_HOST:-127.0.0.1}:${APP_PORT:-3015}:80" in compose
