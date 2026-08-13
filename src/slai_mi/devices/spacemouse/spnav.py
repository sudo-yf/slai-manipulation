"""Small Python 3 ctypes adapter for the system libspnav."""

from __future__ import annotations

from ctypes import CDLL, POINTER, Structure, Union, c_int, c_uint, c_void_p, pointer
from dataclasses import dataclass
from typing import ClassVar

SPNAV_EVENT_MOTION = 1
SPNAV_EVENT_BUTTON = 2


@dataclass(frozen=True)
class SpnavMotionEvent:
    translation: tuple[int, int, int]
    rotation: tuple[int, int, int]
    period: int


@dataclass(frozen=True)
class SpnavButtonEvent:
    bnum: int
    press: bool


class _MotionEvent(Structure):
    _fields_: ClassVar = [
        ("type", c_int),
        ("x", c_int),
        ("y", c_int),
        ("z", c_int),
        ("rx", c_int),
        ("ry", c_int),
        ("rz", c_int),
        ("period", c_uint),
        ("data", c_void_p),
    ]


class _ButtonEvent(Structure):
    _fields_: ClassVar = [("type", c_int), ("press", c_int), ("bnum", c_int)]


class _Event(Union):
    _fields_: ClassVar = [("type", c_int), ("motion", _MotionEvent), ("button", _ButtonEvent)]


_LIB = CDLL("libspnav.so")
_LIB.spnav_open.argtypes = []
_LIB.spnav_open.restype = c_int
_LIB.spnav_close.argtypes = []
_LIB.spnav_close.restype = None
_LIB.spnav_poll_event.argtypes = [POINTER(_Event)]
_LIB.spnav_poll_event.restype = c_int


def open_connection() -> None:
    if _LIB.spnav_open() == -1:
        raise RuntimeError("failed to connect to spacenavd")


def close_connection() -> None:
    _LIB.spnav_close()


def poll_event() -> SpnavMotionEvent | SpnavButtonEvent | None:
    event = _Event()
    if _LIB.spnav_poll_event(pointer(event)) == 0:
        return None
    if event.type == SPNAV_EVENT_MOTION:
        motion = event.motion
        return SpnavMotionEvent(
            translation=(motion.x, motion.y, motion.z),
            rotation=(motion.rx, motion.ry, motion.rz),
            period=int(motion.period),
        )
    if event.type == SPNAV_EVENT_BUTTON:
        button = event.button
        return SpnavButtonEvent(bnum=int(button.bnum), press=bool(button.press))
    raise RuntimeError(f"unknown spnav event type: {event.type}")
