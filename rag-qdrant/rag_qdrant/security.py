"""Security, authentication, and client profile authorization."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from urllib.parse import urlparse

from .config import ADMIN_TOKEN, AMIGA_TOKEN, DEVNOTES_TOKEN


@dataclass
class ClientProfile:
    name: str
    token: str
    can_read: bool = True
    can_write: bool = False
    allowed_search_sources: List[str] = field(default_factory=lambda: ["*"])
    allowed_index_sources: List[str] = field(default_factory=list)
    allowed_index_directories: List[str] = field(default_factory=list)


def _load_profiles() -> Dict[str, ClientProfile]:
    """Build registry of configured client profiles from environment/config."""
    profiles: Dict[str, ClientProfile] = {}

    # 1. Admin profile
    admin_token = ADMIN_TOKEN.strip()
    if admin_token:
        profiles[admin_token] = ClientProfile(
            name="admin",
            token=admin_token,
            can_read=True,
            can_write=True,
            allowed_search_sources=["*"],
            allowed_index_sources=["*"],
            allowed_index_directories=["*"],
        )

    # 2. Amiga profile
    amiga_token = AMIGA_TOKEN.strip()
    if amiga_token:
        profiles[amiga_token] = ClientProfile(
            name="amiga",
            token=amiga_token,
            can_read=True,
            can_write=True,
            allowed_search_sources=["amiga", "devnotes"],
            allowed_index_sources=["amiga"],
            allowed_index_directories=["*"],  # or specific paths if restricted
        )

    # 3. Devnotes profile
    devnotes_token = DEVNOTES_TOKEN.strip()
    if devnotes_token:
        profiles[devnotes_token] = ClientProfile(
            name="devnotes",
            token=devnotes_token,
            can_read=True,
            can_write=True,
            allowed_search_sources=["devnotes"],
            allowed_index_sources=["devnotes"],
            allowed_index_directories=["*"],
        )

    return profiles


ALLOWED_HOSTS: Set[str] = {
    "127.0.0.1",
    "localhost",
    "[::1]",
    "::1",
}


def validate_host_header(host: Optional[str]) -> bool:
    """Validate that the Host header corresponds strictly to loopback addresses."""
    if not host:
        return False
    # Strip port if present
    hostname = host.split(":")[0].strip()
    # Handle bracketed IPv6: [::1]:6335 -> [::1]
    if host.startswith("["):
        hostname = host.split("]")[0] + "]"
    return hostname in ALLOWED_HOSTS


def validate_origin_header(origin: Optional[str]) -> bool:
    """Validate Origin header against DNS rebinding and cross-site requests."""
    if not origin:
        # Non-browser clients (CLI, python scripts, curl) omit Origin
        return True
    try:
        parsed = urlparse(origin)
        hostname = (parsed.hostname or "").strip()
        return hostname in ALLOWED_HOSTS
    except Exception:
        return False


def get_profile_by_token(token: Optional[str]) -> Optional[ClientProfile]:
    """Retrieve ClientProfile by secret token."""
    if not token:
        return None
    token = token.strip()
    profiles = _load_profiles()

    if token in profiles:
        return profiles[token]

    # If no tokens were configured at all in the environment, fallback to a local default profile
    if not profiles and token in ("local-dev-token", "default"):
        return ClientProfile(
            name="local-default",
            token=token,
            can_read=True,
            can_write=True,
            allowed_search_sources=["*"],
            allowed_index_sources=["*"],
            allowed_index_directories=["*"],
        )

    return None


def validate_search_sources(
    requested_sources: Optional[List[str]], profile: ClientProfile
) -> Tuple[Optional[List[str]], Optional[str]]:
    """Verify and constrain search sources according to client profile.

    Returns (effective_sources, error_message).
    """
    allowed = profile.allowed_search_sources

    # If profile allows all sources
    if "*" in allowed:
        return requested_sources, None

    # If client did not specify a source filter, default to allowed profile sources
    if not requested_sources:
        return list(allowed), None

    # Check each requested source
    for s in requested_sources:
        s_clean = s.strip().lower()
        if s_clean not in [a.lower() for a in allowed]:
            return None, f"Source '{s}' is not permitted for client profile '{profile.name}'."

    return requested_sources, None


def validate_index_authorization(
    directory: Path, source: str, profile: ClientProfile
) -> Tuple[bool, Optional[str]]:
    """Verify that the client profile is permitted to index this directory and source."""
    if not profile.can_write:
        return False, f"Profile '{profile.name}' has read-only access and cannot index."

    # Validate source
    if "*" not in profile.allowed_index_sources:
        allowed_sources_lower = [s.lower() for s in profile.allowed_index_sources]
        if source.strip().lower() not in allowed_sources_lower:
            return (
                False,
                f"Source '{source}' is not permitted for profile '{profile.name}' (allowed: {profile.allowed_index_sources}).",
            )

    # Validate directory
    resolved_dir = directory.resolve()
    if not resolved_dir.is_dir():
        return False, f"Target directory '{directory}' does not exist or is not a directory."

    if "*" not in profile.allowed_index_directories:
        allowed = False
        for allowed_root in profile.allowed_index_directories:
            root_path = Path(allowed_root).resolve()
            try:
                if resolved_dir.is_relative_to(root_path):
                    allowed = True
                    break
            except (ValueError, AttributeError):
                if str(resolved_dir).startswith(str(root_path)):
                    allowed = True
                    break
        if not allowed:
            return (
                False,
                f"Directory '{directory}' is outside permitted index directories for profile '{profile.name}'.",
            )

    return True, None
