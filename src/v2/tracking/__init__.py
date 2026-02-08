#!/usr/bin/env python3
"""
Object tracking module for Vision Assistant v2.

Provides multi-object tracking with state machine lifecycle management.
"""

from .state_machine import ObjectStateMachine, StateTransition
from .tracker import MultiObjectTracker

__all__ = [
    "ObjectStateMachine",
    "StateTransition",
    "MultiObjectTracker",
]
