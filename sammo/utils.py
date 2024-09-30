# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Small number of utility functions that are used across SAMMO."""
import asyncio
import collections
import tempfile
import time
import pathlib
import sys
import webbrowser
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

GRAPH_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">

<head>
  <style>
  body,
  html {
    margin: 0;
    height: 100%;
    width: 100%;
  }

  .split {
    display: flex;
    flex-direction: row;
    height: 100%;
    overflow: hidden;
  }
  .gutter {
    background-color: #eee;
    background-repeat: no-repeat;
    background-position: 50%;
  }

  .gutter.gutter-horizontal {
    background-image: url('data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAUAAAAeCAYAAADkftS9AAAAIklEQVQoU2M4c+bMfxAGAgYYmwGrIIiDjrELjpo5aiZeMwF+yNnOs5KSvgAAAABJRU5ErkJggg==');
    cursor: col-resize;
  }
  #info {
	overflow-y: auto;
  }
  #info div {
    font-family: Helvetica, sans-serif;
    font-weight: 600;
    text-transform: uppercase;
    background-color: #d6d8d9;
    color: #616161;
    font-size: 11px;
    padding: 6px;
  }
  pre {
	white-space: pre-wrap;
	padding: 6px;
	background-color: #f9f9f9;
	margin: 0;
	font-size: 12px;
  }
  </style>
  <meta charset="utf-8" />
  <title>Callgraph</title>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/cytoscape/3.26.0/cytoscape.min.js"></script>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/split.js/1.6.0/split.min.js"></script>
</head>

<body>
  <div class="split">
    <div id="info"><pre>Click on node for details.</pre></div>
    <div id="graph"></div>
  </div>
  <script>
  Split(['#info', '#graph'], {
    sizes: [20, 80],
    minSize: [150, 0],
    onDragEnd: resizeCyto
  });
  const DATA = ELEMENTS;

  var cy = cytoscape({
    container: document.getElementById('graph'),
    style: cytoscape.stylesheet().selector('node').style({
      'content': 'data(label)',
	  'background-color': DATA['node-color'] || 'Teal',
	  'border-width': DATA['node-border'] || 0,
    }).selector(':selected').style({'background-color': 'Turquoise'}).selector('edge').style({
      'curve-style': 'bezier',
      'target-arrow-shape': 'triangle',
      'width': 1,
      'line-color': 'black',
      'target-arrow-color': 'black'
    }),
    elements: DATA,
    wheelSensitivity: 0.3,
    layout: {
      name: 'breadthfirst',
      directed: true,
      depthSort: function(a, b){ return a.data('priority') - b.data('priority') }
    }
  });

  function resizeCyto() {
    cy.resize();
    cy.fit();
  }

  function escapeHTML(str){
    return new Option(str).innerHTML;
  }

  window.addEventListener('resize', resizeCyto);
  cy.on('tap', 'node', function(evt) {
    const node = evt.target;
    const details = node.data('details');
    const info = document.getElementById('info');
    info.innerHTML = '';

    if (!details) {
        info.innerHTML = "<div>Node has no metadata.</div>";
    } else if (typeof details === 'object' && !Array.isArray(details)) {
        Object.entries(details).forEach(([key, value]) => {
            info.innerHTML += `<div>${escapeHTML(key)}</div><pre>${escapeHTML(value)}</pre>`;
        });
    } else {
        info.innerHTML = `<pre>${escapeHTML(details)}</pre>`;
    }
  });
  </script>
</body>

</html>
"""


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
    return hasattr(sys, "ps1")


def is_jupyter() -> bool:
    """Check if code is running in jupyter lab or notebook."""
    try:
        if get_ipython().__class__.__name__ == "ZMQInteractiveShell":
            return True
        else:
            return False
    except NameError:
        return False


class HtmlRenderer:
    """Render HTML in an IFrame for Jupyter or as temporary file."""

    def __init__(self, raw_html, width="100%", height="300px"):
        self.raw_html = raw_html
        self.width = width
        self.height = height

    def _repr_html_(self, **kwargs):
        iframe = f"""\
            <iframe srcdoc="{escape(self.raw_html)}" width="{self.width}" height="{self.height}"'
            "allowfullscreen" style="border:1px solid #e0e0e0; box-sizing: border-box;">
            </iframe>"""
        return iframe

    def render(self, backend="auto"):
        if backend == "auto":
            backend = "jupyter" if is_jupyter() else "file"
        if backend == "jupyter":
            return self
        else:
            with tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8", suffix=".html") as f:
                f.write(self.raw_html)
                webbrowser.open("file://" + f.name, new=2, autoraise=True)


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
