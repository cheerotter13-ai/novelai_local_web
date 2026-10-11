import io
import zipfile

import pytest
from PIL import Image

from api_utils.novelai_client import NovelAIClient, NovelAIUpstreamError, resolve_upstream
from app import create_app
from conftest import ORIGIN


class FakeResponse:
    def __init__(self, status_code=200, payload=None, content=b"", headers=None):
        self.status_code = status_code
        self._payload = payload
        self.content = content or (b"" if payload is None else b"{}")
        self.headers = headers or {}

    def json(self):
        if self._payload is None:
            raise ValueError("not json")
        return self._payload


def test_resolve_upstream_rejects_arbitrary_hosts():
    assert resolve_upstream("tkm")[0] == "takoma"
    assert resolve_upstream(None)[2] == "image.novelai.net"
    with pytest.raises(ValueError):
        resolve_upstream("https://evil.example")


def test_takoma_account_snapshot_skips_unsupported_information_route():
    urls = []

    def request_func(method, url, **kwargs):
        urls.append((method, url))
        assert kwargs["headers"]["Authorization"] == "Bearer key"
        return FakeResponse(200, {
            "tier": 3,
            "active": True,
            "trainingStepsLeft": {"fixedTrainingStepsLeft": 10, "purchasedTrainingSteps": 0},
        })

    snapshot = NovelAIClient(request_func=request_func, upstream="takoma").account_snapshot("key")
    assert urls == [("GET", "https://api.takoma.app/user/subscription")]
    assert snapshot["information"] == {}
    assert snapshot["subscription"]["tier"] == 3


def test_takoma_password_login_is_rejected_before_any_request():
    def request_func(*args, **kwargs):
        raise AssertionError("password login must not reach Takoma")

    with pytest.raises(NovelAIUpstreamError) as error:
        NovelAIClient(request_func=request_func, upstream="takoma").login_with_password("a@b.c", "secret")
    assert error.value.code == "TAKOMA_PASSWORD_LOGIN_UNSUPPORTED"


def test_generate_image_accepts_zip_response():
    image = io.BytesIO()
    Image.new("RGB", (2, 2), (1, 2, 3)).save(image, format="PNG")
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("image_0.png", image.getvalue())

    def request_func(method, url, **kwargs):
        assert url == "https://api.takoma.app/ai/generate-image"
        return FakeResponse(200, content=archive.getvalue(), headers={"Content-Type": "application/zip"})

    images = NovelAIClient(request_func=request_func, upstream="takoma").generate_image("key", {"input": "1girl"}, "cid")
    assert images[0]["mime_type"] == "image/png"
    assert images[0]["index"] == 0


def test_upstream_route_and_token_login_use_takoma_client(tmp_path):
    def request_func(method, url, **kwargs):
        assert url == "https://api.takoma.app/user/subscription"
        return FakeResponse(200, {
            "tier": 3,
            "active": True,
            "expiresAt": 1792338720,
            "trainingStepsLeft": {"fixedTrainingStepsLeft": 12, "purchasedTrainingSteps": 1},
        })

    app = create_app(
        {"TESTING": True, "DATA_DIR": str(tmp_path / "data"), "UPSTREAM": "takoma"},
        novelai_client=NovelAIClient(request_func=request_func, upstream="takoma"),
    )
    client = app.test_client()
    profile = client.get("/api/upstream")
    assert profile.status_code == 200
    assert profile.get_json() == {"provider": "takoma", "password_login": False}
    login = client.post(
        "/api/session/persistent-token",
        json={"token": "local-test-key"},
        headers={"Origin": ORIGIN},
    )
    assert login.status_code == 200
    body = login.get_json()
    assert body["authenticated"] is True
    assert body["account_snapshot"]["subscription"]["tier"] == 3
    assert body["account_snapshot"]["anlas"]["total"] == 13
    assert "local-test-key" not in login.get_data(as_text=True)
