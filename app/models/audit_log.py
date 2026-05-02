"""
Audit log model for tracking user actions.
"""

import ipaddress
from typing import Optional

from fastapi import Request
from sqlalchemy import JSON, Column, DateTime, ForeignKey, Integer, String
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from app.database import Base


class AuditLog(Base):
    """Audit log model for tracking system events."""

    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True, index=True)
    action = Column(
        String(100), nullable=False, index=True
    )  # e.g., "login", "create_alert", "export_evidence"
    resource_type = Column(String(50), index=True)  # e.g., "detection", "user", "camera"
    resource_id = Column(Integer, nullable=True)
    ip_address = Column(String(45))
    user_agent = Column(String(500))
    details = Column(JSON)  # Additional action details
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)

    # Relationships
    user = relationship("User", back_populates="audit_logs")


def _normalize_client_ip_string(addr: Optional[str]) -> Optional[str]:
    """Canonical text for IPv4; drop IPv4-mapped IPv6 prefix; clamp DB length (IPv6 max 45 chars)."""
    if not addr:
        return None
    s = addr.strip().strip('"').strip()
    if not s or s.lower() == "unknown":
        return None
    if s.startswith("["):
        bracket_end = s.find("]")
        if bracket_end != -1:
            s = s[1:bracket_end]
    if "%" in s:
        s = s.split("%", 1)[0]
    try:
        ip = ipaddress.ip_address(s)
        if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped:
            s = str(ip.ipv4_mapped)
        else:
            s = str(ip)
    except ValueError:
        pass
    return s[:45] if len(s) > 45 else s


def _first_xff_hop(xff: str) -> Optional[str]:
    for part in xff.split(","):
        c = part.strip()
        if c and c.lower() != "unknown":
            return c
    return None


def _forwarded_strip_port(raw: str) -> str:
    v = raw.strip()
    if v.startswith("["):
        end = v.find("]")
        if end != -1:
            return v[: end + 1]
        return v
    if v.count(":") == 1:
        host, _, maybe_port = v.rpartition(":")
        if maybe_port.isdigit():
            return host
    return v


def _forwarded_first_for(forwarded: str) -> Optional[str]:
    for segment in forwarded.split(","):
        for param in segment.split(";"):
            p = param.strip()
            lp = p.lower()
            if not lp.startswith("for="):
                continue
            val = p.split("=", 1)[1].strip()
            while val.startswith('"') and val.endswith('"') and len(val) >= 2:
                val = val[1:-1].strip()
            if val.startswith("_"):
                continue
            return _forwarded_strip_port(val)
    return None


def get_client_ip(request: Request) -> Optional[str]:
    """
    Best-effort client IP behind reverse proxies.

    Prefer X-Forwarded-For left-most client hop, then RFC 7239 Forwarded ``for=``,
    then X-Real-IP / CF-Connecting-IP / True-Client-IP, then ASGI scope client
    (often populated by ProxyHeadersMiddleware when proxies are trusted).
    """
    h = request.headers

    def _trim(v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        v = v.strip()
        return v if v else None

    xff = _trim(h.get("x-forwarded-for"))
    if xff:
        client = _first_xff_hop(xff)
        if client:
            return _normalize_client_ip_string(client)

    fwd = _trim(h.get("forwarded"))
    if fwd:
        client = _forwarded_first_for(fwd)
        if client:
            return _normalize_client_ip_string(client)

    for hdr in ("x-real-ip", "cf-connecting-ip", "true-client-ip"):
        cand = _trim(h.get(hdr))
        if cand:
            return _normalize_client_ip_string(cand)

    host = getattr(request.client, "host", None)
    return _normalize_client_ip_string(host)


def create_audit_log(
    request: Request,
    user_id: int,
    action: str,
    resource_type: str,
    resource_id: int = None,
    details: dict = None,
) -> AuditLog:
    """Create an AuditLog with IP address and user agent from the request."""
    ip = get_client_ip(request)
    ua = request.headers.get("user-agent")
    return AuditLog(
        user_id=user_id,
        action=action,
        resource_type=resource_type,
        resource_id=resource_id,
        ip_address=ip,
        user_agent=ua,
        details=details,
    )
