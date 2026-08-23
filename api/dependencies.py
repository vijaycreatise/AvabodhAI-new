"""
api/dependencies.py
--------------------
Shared FastAPI dependencies.

get_tenant_id() is the ONE place tenant identity enters the KB container.
Every route that reads or writes documents / chat_threads / chat_messages
(Postgres, RLS-enforced) or Qdrant chunk/chat-message points (isolated
solely via pipeline/retriever.py::build_filter(), no RLS equivalent there)
depends on this and threads the value all the way down — there is no
other mechanism enforcing isolation between tenants.

Trust boundary: this header is trusted as coming from the Security Gateway
(or an equivalent internal caller) — NOT from the end user directly. That
only holds if the KB container is unreachable except through that trusted
caller (internal network / service mesh / mTLS). If the KB container is
ever exposed directly to end users, this header must instead be derived
from a verified token (JWT claim, session lookup, etc.), not trusted as
plain input.
"""

import re

from fastapi import Header, HTTPException

_MAX_TENANT_ID_LENGTH = 128
_MAX_ORG_UNIT_ID_LENGTH = 128

# Characters that make an identifier unsafe to use as a filesystem path
# component or an object-storage key segment. Both tenant_id and org_unit_id
# are used that way (api/routes/documents.py::_tenant_upload_dir,
# pipeline/object_store.py::build_key), so a value like "../../etc" would
# otherwise escape the directory it is supposed to scope.
#
# A denylist rather than an allowlist on purpose: these ids come from a
# gateway and their format is that caller's choice (today they are UUIDs),
# so rejecting only what is genuinely dangerous avoids breaking a legitimate
# id shape we did not anticipate.
_UNSAFE_ID_PATTERN = re.compile("[" + re.escape("/" + chr(92) + chr(0)) + "]|" + re.escape(".."))


def _reject_unsafe_id(value: str, header_name: str) -> None:
    if _UNSAFE_ID_PATTERN.search(value):
        raise HTTPException(
            status_code=400,
            detail=f"{header_name} contains characters that are not allowed",
        )



def get_tenant_id(x_tenant_id: str = Header(..., alias="X-Tenant-ID")) -> str:
    """
    FastAPI dependency — every tenant-scoped route takes this as a parameter:

        tenant_id: str = Depends(get_tenant_id)

    Missing, blank, or oversized header -> 400, rejected before it ever
    reaches the database layer.
    """
    tenant_id = (x_tenant_id or "").strip()
    if not tenant_id:
        raise HTTPException(status_code=400, detail="X-Tenant-ID header is required")
    if len(tenant_id) > _MAX_TENANT_ID_LENGTH:
        raise HTTPException(status_code=400, detail="X-Tenant-ID header is too long")
    _reject_unsafe_id(tenant_id, "X-Tenant-ID")
    return tenant_id


def get_org_unit_id(x_org_unit_id: str = Header(..., alias="X-Org-Unit-ID")) -> str:
    """
    FastAPI dependency — second, required isolation identifier, same
    trust model as get_tenant_id: sourced from a header set by the
    frontend/gateway from the logged-in user's session, never typed by
    the user directly. Every route that also depends on get_tenant_id
    must depend on this too — org_unit_id is a hard boundary WITHIN a
    tenant (department-level), always applied together with tenant_id,
    never filtered on its own (see db/models.py composite indexes —
    a standalone org_unit_id filter would let a department code that
    happens to collide across two different tenants leak across
    companies, not just across departments).
    """
    org_unit_id = (x_org_unit_id or "").strip()
    if not org_unit_id:
        raise HTTPException(status_code=400, detail="X-Org-Unit-ID header is required")
    if len(org_unit_id) > _MAX_ORG_UNIT_ID_LENGTH:
        raise HTTPException(status_code=400, detail="X-Org-Unit-ID header is too long")
    _reject_unsafe_id(org_unit_id, "X-Org-Unit-ID")
    return org_unit_id