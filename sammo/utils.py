# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Small number of utility functions that are used across SAMMO."""
import asyncio
import collections
import time
import pathlib
import sys
from concurrent.futures import ThreadPoolExecutor
from html import escape

from orjson import orjson

__all__ = [
    "CodeTimer",
    "MAIN_PATH",
    "MAIN_NAME",
    "DEFAULT_SAVE_PATH",
    "sync",
    "serialize_json",
]


class CodeTimer:
    """Time code with this context manager."""

    def __init__(self):
        self.created = time.perf_counter()
        self._interval = None

    @property
    def interval(self) -> float:
        """Timed interval in s."""
        if self._interval is None:
            return time.perf_counter() - self.created
        return self._interval

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.end = time.perf_counter()
        self._interval = self.end - self.start


def is_thread_running_async_loop() -> bool:
    try:
        asyncio.get_running_loop()
        return True
    except RuntimeError:
        return False


def sync(f: collections.abc.Coroutine):
    """Execute and return result of an async function. Take special care of already running async loops."""
    if is_thread_running_async_loop():
        # run inside a new thread
        with ThreadPoolExecutor(1) as pool:
            result = pool.submit(lambda: asyncio.run(f))
        return result.result()
    else:
        return asyncio.run(f)


def is_interactive() -> bool:
    """Check if the code is running in an interactive shell."""
    try:
        get_ipython()
        return True
    except:
        return False


class IFrameRenderer:
    """Render HTML in an IFrame for Jupyter Lab and Notebook"""

    def __init__(self, raw_html, width="100%", height="300px"):
        self.raw_html = raw_html
        self.width = width
        self.height = height

    def _repr_html_(self, **kwargs):
        iframe = f"""\
            <iframe srcdoc="{escape(self.raw_html)}" width="{self.width}" height="{self.height}"'
            "allowfullscreen" style="border:1px solid #e0e0e0;">
            </iframe>"""
        return iframe


def get_main_script_path() -> pathlib.Path:
    """Path of the main script if not interactive, otherwise working dir."""
    if is_interactive():
        return pathlib.Path.cwd().resolve()
    else:
        return pathlib.Path(sys.argv[0]).resolve().parent


def get_main_script_name(if_interactive="tmp") -> str:
    """Name of the main script file if not interactive, otherwise 'tmp'."""
    if is_interactive():
        return if_interactive
    else:
        return pathlib.Path(sys.argv[0]).name


def get_default_save_path() -> str:
    """Default save path is folder with the same name as main script."""
    if is_interactive():
        return get_main_script_path() / get_main_script_name()
    else:
        return pathlib.Path(sys.argv[0]).with_suffix("").resolve()


def serialize_json(key) -> bytes:
    """Serialize json with orjson to invariant byte string."""
    return orjson.dumps(key, option=orjson.OPT_SORT_KEYS)


MAIN_PATH = get_main_script_path()
"""Path of the main script if not interactive, otherwise working dir."""

MAIN_NAME = get_main_script_name()
"""Name of the main script file if not interactive, otherwise 'tmp'."""

DEFAULT_SAVE_PATH = get_default_save_path()
"""Default save path is folder with the same name as main script."""
