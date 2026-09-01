"""
api/middleware/logging_middleware.py
------------------------------------
Logs every request and response — method, path, status, time taken.

Phase H #8 (2026-08-21): generates/propagates X-Request-ID for log
correlation. If the caller sends X-Request-ID, that value is reused
(lets an upstream system like clariona-core thread its own trace ID
through); otherwise one is generated. Stashed on request.state so other
code in the request (route handlers, pipeline calls) can log it too if
they choose to, and echoed back as a response header either way — purely
additive, no existing response header changes.
"""

import time
import uuid
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from utils.logger import get_logger

logger = get_logger(__name__)


class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
        request.state.request_id = request_id

        start = time.time()
        response = await call_next(request)
        elapsed = round((time.time() - start) * 1000, 2)

        response.headers["X-Request-ID"] = request_id

        # Logged whenever present — not enforced here (that's
        # api.dependencies.get_tenant_id/get_org_unit_id's job), just
        # surfaced so a tenant/department-specific incident can be traced
        # straight from the logs.
        tenant_id = request.headers.get("x-tenant-id", "-")
        org_unit_id = request.headers.get("x-org-unit-id", "-")
        logger.info(
            "[%s] %s %s -> %d | %.2fms | tenant=%s org_unit=%s",
            request_id,
            request.method,
            request.url.path,
            response.status_code,
            elapsed,
            tenant_id,
            org_unit_id,
        )
        return response
