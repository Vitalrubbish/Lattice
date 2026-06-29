"""Low-overhead host-side section timing for KCMM diagnostic gates."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any


@dataclass
class _SectionTiming:
    count: int = 0
    total_ns: int = 0
    min_ns: int | None = None
    max_ns: int | None = None

    def add(self, elapsed_ns: int) -> None:
        self.count += 1
        self.total_ns += elapsed_ns
        self.min_ns = (
            elapsed_ns if self.min_ns is None else min(self.min_ns, elapsed_ns)
        )
        self.max_ns = (
            elapsed_ns if self.max_ns is None else max(self.max_ns, elapsed_ns)
        )


class HostSectionProfiler:
    """Accumulates section-level wall-clock timings without storing samples."""

    def __init__(self, enabled: bool = False) -> None:
        self.enabled = bool(enabled)
        self._sections: dict[str, _SectionTiming] = {}

    def start(self) -> int:
        return time.perf_counter_ns() if self.enabled else 0

    def stop(self, section: str, started_ns: int) -> None:
        if not self.enabled:
            return
        elapsed_ns = time.perf_counter_ns() - started_ns
        timing = self._sections.setdefault(section, _SectionTiming())
        timing.add(elapsed_ns)

    @staticmethod
    def _rounded_ms(value_ns: int | None) -> float | None:
        return round(value_ns / 1_000_000, 6) if value_ns is not None else None

    @staticmethod
    def _rounded_us(value_ns: int | None) -> float | None:
        return round(value_ns / 1_000, 3) if value_ns is not None else None

    def report(self) -> dict[str, Any]:
        sections: dict[str, Any] = {}
        for name, timing in sorted(self._sections.items()):
            avg_ns = timing.total_ns / timing.count if timing.count else None
            sections[name] = {
                "count": timing.count,
                "total_ms": self._rounded_ms(timing.total_ns),
                "avg_us": self._rounded_us(int(avg_ns)) if avg_ns is not None else None,
                "min_us": self._rounded_us(timing.min_ns),
                "max_us": self._rounded_us(timing.max_ns),
            }
        return {
            "enabled": self.enabled,
            "unit": "wall_clock",
            "sections": sections,
        }
