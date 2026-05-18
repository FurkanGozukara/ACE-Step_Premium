"""Out-of-band cancellation route for Gradio generation runs."""

from __future__ import annotations

from typing import Any

from fastapi.responses import JSONResponse
from loguru import logger

from acestep.core.generation.cancellation import request_generation_cancel


CANCEL_GENERATION_ENDPOINT = "/ace-step/cancel-generation"


def register_generation_cancel_route(demo: Any) -> None:
    """Register a non-queued HTTP route that requests generation cancellation."""

    app = getattr(demo, "app", None)
    if app is None or getattr(app, "_ace_cancel_route_registered", False):
        return

    @app.post(CANCEL_GENERATION_ENDPOINT)
    async def _cancel_generation() -> JSONResponse:
        """Request cancellation without going through the Gradio event queue."""

        had_active_work = request_generation_cancel(subprocess_only=True)
        if had_active_work:
            logger.info("[generation_cancel] Cancellation requested through HTTP endpoint.")
            return JSONResponse(
                {
                    "active": True,
                    "status": (
                        "Subprocess cancellation requested. "
                        "The isolated worker is being stopped."
                    ),
                }
            )
        logger.info("[generation_cancel] HTTP cancel requested, but no subprocess is active.")
        return JSONResponse(
            {
                "active": False,
                "status": "No active subprocess generation is currently running.",
            }
        )

    app._ace_cancel_route_registered = True
