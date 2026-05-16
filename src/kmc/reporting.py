"""Progress reporting for KMC operations.

Provides a simple progress reporter that works without external dependencies.
Optionally uses 'rich' for nicer output if available, but gracefully degrades
to plain text output when rich is not installed.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field


@dataclass
class ProgressReporter:
    """Reports progress during pack, unpack, and verify operations.

    When json_mode is True, no progress output is produced (to avoid
    mixing with JSON output).

    Usage:
        reporter = ProgressReporter(total_blocks=500, json_mode=False)
        reporter.start("Packing")
        for i in range(500):
            ...  # do work
            reporter.update(i + 1)
        reporter.finish()
    """

    total_blocks: int = 0
    json_mode: bool = False
    show_progress: bool = True
    operation: str = ""
    _current: int = field(default=0, init=False, repr=False)
    _last_reported_pct: int = field(default=-1, init=False, repr=False)

    def start(self, operation: str = "") -> None:
        """Start progress reporting.

        Args:
            operation: Description of the operation (e.g., 'Packing', 'Unpacking').
        """
        if not self.show_progress or self.json_mode:
            return
        self.operation = operation or self.operation
        if self.total_blocks > 0:
            sys.stderr.write(f"{self.operation}: {self.total_blocks} blocks\n")
        else:
            sys.stderr.write(f"{self.operation}...\n")
        sys.stderr.flush()

    def update(self, current: int, message: str = "") -> None:
        """Update progress.

        Reports progress at approximately 10% intervals to avoid
        excessive output.

        Args:
            current: Current number of completed blocks.
            message: Optional additional message.
        """
        if not self.show_progress or self.json_mode:
            return
        self._current = current

        if self.total_blocks > 0:
            pct = int(100 * current / self.total_blocks)
            # Report at ~10% intervals
            if pct >= self._last_reported_pct + 10 or current == self.total_blocks:
                sys.stderr.write(f"  Processed: {current}/{self.total_blocks} blocks ({pct}%)\n")
                sys.stderr.flush()
                self._last_reported_pct = pct
        else:
            # No total known, report every 100 blocks
            if current % 100 == 0 or message:
                msg = f"  Processed: {current} blocks"
                if message:
                    msg += f" — {message}"
                sys.stderr.write(msg + "\n")
                sys.stderr.flush()

    def finish(self, message: str = "") -> None:
        """Finish progress reporting.

        Args:
            message: Optional final message (e.g., 'Done in 2.5s').
        """
        if not self.show_progress or self.json_mode:
            return
        msg = f"{self.operation}: complete"
        if self._current > 0 and self.total_blocks > 0:
            msg += f" ({self._current}/{self.total_blocks} blocks)"
        if message:
            msg += f" — {message}"
        sys.stderr.write(msg + "\n")
        sys.stderr.flush()


def create_reporter(
    show_progress: bool = False,
    json_mode: bool = False,
    total_blocks: int = 0,
) -> ProgressReporter:
    """Create a ProgressReporter with the given settings.

    Args:
        show_progress: Whether to show progress output.
        json_mode: If True, suppress all progress output.
        total_blocks: Total number of blocks expected.

    Returns:
        Configured ProgressReporter instance.
    """
    return ProgressReporter(
        total_blocks=total_blocks,
        json_mode=json_mode,
        show_progress=show_progress,
    )
