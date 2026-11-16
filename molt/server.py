"""molt-core HTTP server.

Only the health endpoint and the static web app exist so far. Chat, session, squeeze, profile, and
telemetry endpoints arrive in Task 11.

Run from the repository root:  uv run python -m molt.server
"""

from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from molt import __version__, defaults

WEB_DIST = Path(__file__).resolve().parent.parent / "web" / "dist"


def create_app(web_dist: Path | None = WEB_DIST) -> FastAPI:
    app = FastAPI(title="Molt", version=__version__)

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "version": __version__}

    # Mounted last so that it never shadows an API route. Absent until `npm run build`.
    if web_dist is not None and web_dist.is_dir():
        app.mount("/", StaticFiles(directory=web_dist, html=True), name="web")
    return app


app = create_app()


def main() -> None:
    uvicorn.run(app, host=defaults.SERVER_HOST, port=defaults.SERVER_PORT)


if __name__ == "__main__":
    main()
