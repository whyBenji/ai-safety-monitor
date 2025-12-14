"""
Legacy moderation module - kept for backward compatibility.
For new projects, use monitor.pipeline instead.
"""

from monitor.moderator.moderation_service import ModerationService

__all__ = ["ModerationService"]
