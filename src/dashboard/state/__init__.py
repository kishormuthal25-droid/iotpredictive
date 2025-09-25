"""
Dashboard State Management Module
Shared state management for dashboard components
"""

from .shared_state import shared_state_manager
from .time_state import time_state_manager

__all__ = [
    'shared_state_manager',
    'time_state_manager'
]