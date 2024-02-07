# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
Provides a way of displaying multiple progress bars in a single line. Works in both interactive and non-interactive
environments.
"""
import collections
import datetime
import io
import math
import shutil
import sys
import time
from beartype import beartype

from sammo import utils

__all__ = ["CompactProgressBars", "SubProgressBar"]


@beartype
class LinePrinter:
    """
    A class that prints a line to a text output.

    :param out: The output device to print to.
    """

    BACKSPACE = "\b" * 1000

    def __init__(self, out: io.TextIOBase = sys.stdout):
        self._out = out
        self._clear_prefix = ""
        self._is_interactive = utils.is_interactive()
        self._is_finalized = False

    @staticmethod
    def get_terminal_width(default_width: int = 120) -> int:
        width, _ = shutil.get_terminal_size(fallback=(default_width, 80))
        return width

    def print(self, value: str):
        print(self._clear_prefix + value, file=self._out, flush=True, end="")
        if self._is_interactive:
            self._clear_prefix = "\r"
        else:
            self._clear_prefix = self.BACKSPACE[: len(value)]

    def finalize(self):
        if not self._is_finalized:
            print(file=self._out, flush=True)
            self._is_finalized = True


@beartype
class SubProgressBar:
    """
    A class that represents an individual progress bar.

    :param total: The total number of items to process.
    :param  parent: The parent progress bar.
    :param moving_avg_size: The size of the moving average window for calculating the rate.
    :param width: The width of the progress bar in characters.
    :param prefix: The prefix to display before the progress bar.
    :param show_rate: Whether to show the rate of progress.
    :param show_time: Whether to show the elapsed time and ETA.
    :param ascii: Whether to use ASCII (or UTF-8) characters for the progress bar. If "auto", uses ASCII if pdb is imported.
    """

    phases = {True: (" ", "_", "*", "#"), False: (" ", "▏", "▎", "▍", "▌", "▋", "▊", "▉", "█")}

    def __init__(
        self,
        total: int,
        parent: "CompactProgressBars",
        moving_avg_size: int = 10,
        width: int = 100,
        prefix: str = "",
        show_rate: bool = True,
        show_time: bool = True,
        ascii: str = "auto",
    ):
        self._start = time.monotonic()
        self._now = time.monotonic()
        self._last_updates = collections.deque(maxlen=moving_avg_size)
        self._last_updates.appendleft(self._now)
        if ascii == "auto":
            ascii = "pdb" in sys.modules  # Use ascii if debugging
        self.phases = SubProgressBar.phases[ascii]
        self.total = total
        self._n_done = 0
        self._prefix = prefix
        self._parent = parent
        self._show_time = show_time
        self._width = width
        self.max_width = width
        self._show_rate = show_rate

    @property
    def total(self):
        return self._total

    @total.setter
    def total(self, value):
        if value <= 0:
            raise ValueError("total must be positive")
        self._total = value

    @property
    def done(self):
        return self._n_done == self._total

    @property
    def elapsed_long(self):
        return datetime.timedelta(seconds=int(self._now - self._start))

    @property
    def elapsed(self):
        return self._shorten(self.elapsed_long)

    @classmethod
    def _shorten(cls, val: datetime.timedelta):
        """
        Shortens a timedelta object representation by removing leading hours.
        """
        if val.total_seconds() < 3600:
            return str(val)[2:]
        else:
            return val

    @property
    def phase(self):
        remainder = (self._width * self._n_done / self._total) % 1
        phase_index = int(round((len(self.phases) - 1) * remainder))
        if remainder == 0:
            return ""
        else:
            return self.phases[phase_index]

    @property
    def barwidth(self):
        return int(self._width * self._n_done / self._total)

    @property
    def rate(self):
        if self._now == self._start:
            return 0
        elif self._last_updates[0] == self._last_updates[-1]:
            return 1 / (self._now - self._start)
        else:
            return (len(self._last_updates) - 1) / (self._last_updates[0] - self._last_updates[-1])

    @property
    def eta(self):
        if self.rate > 0:
            return self._shorten(datetime.timedelta(seconds=math.ceil((self._total - self._n_done) / self.rate)))
        else:
            return "??:??"

    def update(self, *args, **kwargs):
        """
        Increases the number of completed tasks by one for the progress bar.
        """
        self._n_done += 1
        self._now = time.monotonic()
        self._last_updates.appendleft(self._now)
        self._parent._refresh_display(force=self._n_done == self._total)

    def __str__(self):
        rate = ""
        time = ""
        if self._show_time:
            if self._show_rate:
                rate = f", {self.rate:.2f}it/s"
            time = f"[{self.elapsed}<{self.eta}{rate}]"

        template = f"{self._prefix}{{x}}{self._n_done}/{self._total}{time}"
        self._width = max(5, self.max_width - (len(template) - 3))
        return template.format(
            x=f"[{self.phases[-1] * self.barwidth}"
            f"{self.phase}{(self.phases[0] * (self._width - self.barwidth - len(self.phase)))}]"
        )


@beartype
class CompactProgressBars:
    """
    A class that represents a set of progress bars drawn next to each in a single line.

    :param width: The total width of the progress bar layout in characters.
    :param refresh_interval: The minimum time interval between display refreshes.
    """

    def __init__(self, width: int | None = None, refresh_interval: float = 1 / 50):
        self._bars = collections.OrderedDict()
        self._printer = LinePrinter()
        self._last_update = 0
        self._refresh_interval = refresh_interval

        if width is None:
            self._width = self._printer.get_terminal_width()
        else:
            self._width = width

    def _refresh_display(self, force: bool = False):
        if self._should_refresh() or force:
            self._printer.print(str(self))
            if self._bars and list(self._bars.values())[0].done:
                self.finalize()

    def _should_refresh(self) -> bool:
        if time.monotonic() - self._last_update > self._refresh_interval:
            self._last_update = time.monotonic()
            return True
        return False

    def get(
        self, id: str, total: int | None = None, position: int | None = None, display_name: str | None = None, **kwargs
    ) -> SubProgressBar:
        """
        Gets existing or creates a new progress bar given an id.

        :param id: The id of the progress bar for later reference.
        :param total: Number of increments.
        :param position: Truncate existing bars beyond index and insert this one at the position.
        :param display_name: The name to display for the progress bar. Defaults to id.
        :param **kwargs: Additional arguments to pass to the SubProgressbar constructor.
        :return New bar if it doesn't exist, otherwise a reference to the existing one.
        """
        if id in self._bars:
            existing_bar = self._bars[id]
            existing_bar._total = total
            return existing_bar

        if position is not None:
            self._bars = {k: v for i, (k, v) in enumerate(self._bars.items()) if i < position}
        new_width = self._width // (len(self._bars) + 1)
        for bar in self._bars.values():
            bar.max_width = new_width
        if display_name is None:
            display_name = id
        new_bar = SubProgressBar(total, parent=self, width=new_width, prefix=display_name, **kwargs)
        self._bars[id] = new_bar
        self._refresh_display(True)
        return new_bar

    def finalize(self) -> None:
        """Finishes the line and moves the cursor to the next line."""
        self._printer.finalize()

    def __str__(self) -> str:
        return " >> ".join([str(b) for b in self._bars.values()])
