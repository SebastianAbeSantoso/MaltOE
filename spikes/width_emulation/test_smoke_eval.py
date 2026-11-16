"""Checks smoke_eval.py against a mock OpenAI-compatible server that answers with the datasets' own
gold solutions (expect all solved) or with wrong answers (expect none solved).

Run:  uv run --with pytest pytest -q test_smoke_eval.py      (needs network once, to fetch datasets)
"""
import json
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

import smoke_eval as se

HERE = Path(__file__).parent
CACHE = HERE / "work" / "datasets"


@pytest.fixture(scope="module")
def data():
    return se.load_humaneval(CACHE), se.load_gsm8k(CACHE)


def serve(mode, he, gsm):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            self.send_response(200); self.end_headers(); self.wfile.write(b'{"status":"ok"}')

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            assert body["chat_template_kwargs"] == {"enable_thinking": False} and body["temperature"] == 0
            q = body["messages"][0]["content"]
            reply = "I do not know."
            for p in he:
                if p["prompt"] in q:
                    reply = f"```python\n{p['prompt']}{p['canonical_solution']}\n```" if mode == "oracle" else "```python\ndef nothing():\n    pass\n```"
            for p in gsm:
                if p["question"] in q:
                    gold = se.gsm8k_gold(p["answer"])
                    reply = f"<think>\n\n</think>\nThe answer is\n#### {int(gold):,}" if mode == "oracle" else "#### 424242"
            out = json.dumps({"choices": [{"message": {"content": reply}}], "usage": {"completion_tokens": 7}}).encode()
            self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers()
            self.wfile.write(out)

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


@pytest.mark.parametrize("mode,expect", [("oracle", 5), ("wrong", 0)])
def test_end_to_end_with_mock_server(tmp_path, data, mode, expect):
    he, gsm = data
    srv = serve(mode, he, gsm)
    try:
        out = tmp_path / "r.json"
        r = subprocess.run([sys.executable, str(HERE / "smoke_eval.py"), "--server", f"http://127.0.0.1:{srv.server_port}",
                            "--out", str(out), "--cache", str(CACHE), "--wait", "10"], capture_output=True, text=True)
        assert r.returncode == 0, r.stdout + r.stderr
        s = json.loads(out.read_text())
        assert s["coding"] == {"solved": expect, "n": 5, "errors": 0}
        assert s["reasoning"] == {"solved": expect, "n": 5, "errors": 0}
        assert len(s["draw"]["humaneval_indices"]) == 20 and len(s["draw"]["gsm8k_indices"]) == 20
    finally:
        srv.shutdown()


def test_draw_is_deterministic_and_prefix_stable():
    a, b = se.draw(164, salt="humaneval"), se.draw(164, salt="humaneval")
    assert a == b and len(set(a)) == 20


def test_number_extraction():
    assert se.final_number("so we get 1,234 total.\n#### 1,234") == 1234
    assert se.final_number("The result is 18.") == 18
    assert se.final_number("#### -7") == -7
    assert se.final_number("no numbers here") is None
