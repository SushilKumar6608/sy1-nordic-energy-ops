"""
Typed contracts for tool inputs and outputs.
Wraps TOOL_PERMISSIONS with structured validation helpers.
"""

from src.policy.engine import TOOL_PERMISSIONS, PermissionTier


def get_tool_tier(tool_name: str) -> PermissionTier:
    """Return the PermissionTier for a registered tool. Raises KeyError if unknown."""
    return TOOL_PERMISSIONS[tool_name]


def is_registered(tool_name: str) -> bool:
    """Return True if the tool is registered in TOOL_PERMISSIONS."""
    return tool_name in TOOL_PERMISSIONS


def all_tools() -> list[str]:
    """Return all registered tool names."""
    return list(TOOL_PERMISSIONS.keys())


def tools_by_tier(tier: PermissionTier) -> list[str]:
    """Return all tool names matching the given tier."""
    return [name for name, t in TOOL_PERMISSIONS.items() if t == tier]
