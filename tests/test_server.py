from fastapi.testclient import TestClient

from maltoe import __version__, defaults
from maltoe.server import create_app


def test_health_endpoint():
    client = TestClient(create_app(web_dist=None))
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "version": __version__}


def test_server_listens_on_loopback_only():
    # Req 8.3: prompts and outputs never leave the machine.
    assert defaults.SERVER_HOST == "127.0.0.1"


def test_static_app_is_served_without_shadowing_the_api(tmp_path):
    (tmp_path / "index.html").write_text("<html>maltoe</html>", encoding="utf-8")
    client = TestClient(create_app(web_dist=tmp_path))
    assert client.get("/").text == "<html>maltoe</html>"
    assert client.get("/api/health").json()["status"] == "ok"
