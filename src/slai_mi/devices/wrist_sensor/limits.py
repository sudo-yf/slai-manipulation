"""Measured OpenRB wrist limits from the original platform."""

from __future__ import annotations

import math
from dataclasses import dataclass

RAW_COUNTS_PER_REVOLUTION = 4096


@dataclass(frozen=True)
class JointAngleLimits:
    lower_deg: float
    upper_deg: float

    @classmethod
    def from_raw(cls, lower_raw: int, upper_raw: int, zero_raw: int) -> JointAngleLimits:
        scale = 360.0 / RAW_COUNTS_PER_REVOLUTION
        return cls((lower_raw - zero_raw) * scale, (upper_raw - zero_raw) * scale)

    def inset(self, clearance_deg: float) -> JointAngleLimits:
        result = JointAngleLimits(self.lower_deg + clearance_deg, self.upper_deg - clearance_deg)
        if clearance_deg < 0 or result.lower_deg >= result.upper_deg:
            raise ValueError("invalid joint-limit clearance")
        return result

    @property
    def radians(self) -> tuple[float, float]:
        return math.radians(self.lower_deg), math.radians(self.upper_deg)


RU_MECHANICAL_LIMITS = JointAngleLimits.from_raw(3225, 3876, 3543)
FE_MECHANICAL_LIMITS = JointAngleLimits.from_raw(1758, 3289, 2552)
CONTROL_LIMITS_FE_RU = (FE_MECHANICAL_LIMITS.inset(2.0), RU_MECHANICAL_LIMITS.inset(2.0))
