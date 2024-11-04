# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations
import abc
import copy
import json

from beartype.typing import Callable, Any
from frozendict import frozendict
import pyglove as pg
import pybars
from pyglove import Symbolic
from tabulate import tabulate

from sammo.css_matching import XmlTree
from sammo.utils import HtmlRenderer, GRAPH_TEMPLATE

# monkey-patch pybars to disable HTML escaping
pybars.Compiler._builder.add_escaped_expand = pybars.Compiler._builder.add_expand


class Costs:
    __slots__ = "input", "output"

    def __init__(self, input_costs=0, output_costs=0):
        self.input = input_costs
        self.output = output_costs

    @property
    def total(self):
        return self.input + self.output

    def __add__(self, other):
        return Costs(self.input + other.input, self.output + other.output)

    def __sub__(self, other):
        return Costs(self.input - other.input, self.output - other.output)

    def __repr__(self):
        return f"Costs(input={self.input}, output={self.output})"

    def to_dict(self):
        return {"input": self.input, "output": self.output}


@pg.symbolize
class Runner:
    def __init__(self):
        super().__init__()
        self._costs = Costs()

    def reset_costs(self):
        self._costs = Costs()

    @property
    def costs(self):
        return self._costs


class Result:
    __slots__ = "parent", "value", "stored_values", "op"

    def __init__(self, value, parent=None, stored_values=None, op=None):
        self.value = value
        self.parent = parent
        self.stored_values = stored_values
        self.op = op

    def to_json(self):
        return self.value

    @classmethod
    def bfs(cls, start, match_condition: Callable | None = None):
        """Breadth-first search returning all nodes that match the given condition.

        Args:
            match_condition: A function that returns True if a node matches."""
        queue = [cls(None, parent=start)]
        matches = list()
        while queue:
            node = queue.pop(0)
            if match_condition is None or match_condition(node):
                matches.append(node)
            if isinstance(node, Result):
                queue.extend(node.parents)
        return matches

    def with_parent(self, parent):
        self.parent = parent
        return self

    def with_op(self, op):
        self.op = op
        return self

    def clone_with_stored_value(self, name, value):
        cloned = copy.copy(self)
        if cloned.stored_values is None:
            cloned.stored_values = dict(name=value)
        else:
            cloned.stored_values[name] = value
        return cloned

    @property
    def parents(self):
        if self.parent is None:
            return []
        elif isinstance(self.parent, list):
            return self.parent
        elif isinstance(self.parent, dict):
            return list(self.parent.values())
        else:
            return [self.parent]

    def __repr__(self):
        value_str = repr(self.value)
        if len(value_str) > 100:
            value_str = value_str[:100] + "..."
        return f"{self.__class__.__name__}(value={value_str}, parent={self.parent.__class__.__name__})"

    @staticmethod
    def get_value(result, as_list=False):
        value = result.value if isinstance(result, Result) else result
        if as_list:
            return value if isinstance(value, list) else [value]
        return value

    def values_as_list(self):
        if self.value is None:
            return []
        elif isinstance(self.value, list):
            return self.value
        else:
            return [self.value]

    def plot_call_trace(self, backend="auto"):
        queue = [self]
        nodes = list()
        edges = list()
        while queue:
            node = queue.pop(0)
            is_operator = hasattr(node, "op") and node.op
            node_data = {
                "id": id(node),
                "label": node.op.__class__.__name__ if is_operator else node.__class__.__name__,
                "priority": len(nodes),
                "details": {
                    "Output": node.value if hasattr(node, "value") else str(node),
                    "Parameters": node.op.to_short_string(max_depth=1, include_root=False) if is_operator else "",
                },
            }
            nodes.append({"data": node_data})
            if isinstance(node, Result):
                for parent in node.parents:
                    if not isinstance(parent, Result):
                        continue
                    queue.append(parent)
                    edges.append({"data": {"target": id(node), "source": id(parent)}})
        graph = json.dumps({"nodes": nodes, "edges": edges}, ensure_ascii=False)
        html = GRAPH_TEMPLATE.replace("ELEMENTS", graph)
        return HtmlRenderer(html).render(backend)


class NonEmptyResult(Result):
    pass


class TextResult(NonEmptyResult):
    pass


class LLMResult(NonEmptyResult):
    __slots__ = (
        "value",
        "parent",
        "stored_values" "extra_data",
        "_costs",
        "history",
        "retries",
        "request_text",
        "fingerprint",
    )

    def __init__(
        self,
        value,
        parent=None,
        stored_values=None,
        extra_data=None,
        history=None,
        retries=0,
        costs=None,
        request_text=None,
        fingerprint=None,
    ):
        super().__init__(value, parent, stored_values)
        self.extra_data = extra_data
        self._costs = costs
        self.retries = retries
        self.history = history
        self.request_text = request_text
        self.fingerprint = fingerprint

    @property
    def costs(self):
        return self._costs or Costs()


class ParseResult(NonEmptyResult):
    pass


class EmptyResult(Result):
    def __init__(self, value=None, parent=None, stored_values=None, op=None):
        super().__init__(value, parent, stored_values, op=None)


class TimeoutResult(EmptyResult):
    pass


class EvaluationScore:
    __slot__ = "score", "mistakes", "details"

    def __init__(self, score, mistakes=None, details=None):
        self.score = score
        self.mistakes = mistakes or list()
        self.details = details or dict()

    def to_dict(self, name_score="score"):
        return {name_score: self.score, "mistakes": self.mistakes, "details": self.details}

    def __repr__(self):
        return tabulate(
            [{"name": "score", "value": self.score}] + [{"name": k, "value": v} for k, v in self.details.items()],
            headers="keys",
            maxcolwidths=50,
        )


class PyGloveMatch:
    __slots__ = ["node", "path"]

    def __init__(self, node, path):
        self.node = node
        self.path = path

    def __repr__(self):
        return str(self)

    def __str__(self):
        return f"PyGloveMatch(\n  node={str(self.node)[:50]}..., \n  path={self.path}\n)"


@pg.symbolize(eq=True)
class Component:
    """Base class for all components.

    Components are the building blocks of a search space.

    Args:
        child: Child component. This can be another component or a string.
        reference_id: Id for later querying.
        reference_classes: List of classes that this component belongs to. This is used for querying.
    """

    NEEDS_SCHEDULING = False

    def __init__(self, child: Any | str, reference_id: str | None = None, reference_classes: list[str] | None = None):
        self._id = reference_id
        self._classes = reference_classes

        # auto convert strings
        if isinstance(child, (list, tuple)):
            self._child = [self._ensure_component(c) for c in child]
        else:
            self._child = self._ensure_component(child)
        self.dependencies = list()

    def _ensure_component(self, child):
        if not isinstance(child, Component) and not isinstance(self, Template):
            child = VerbatimText(child)
        return child

    @classmethod
    def _unwrap_results(cls, v):
        if isinstance(v, NonEmptyResult):
            return v.value
        elif isinstance(v, list):
            return [cls._unwrap_results(w) for w in v]
        return v

    def find_first(self, css_selector: str) -> PyGloveMatch:
        matches = self.find_all(css_selector)
        return matches[0] if matches else None

    def find_all(self, css_selector: str) -> list[PyGloveMatch]:
        paths = XmlTree.from_pyglove(self).find_all(css_selector)
        return [PyGloveMatch(node=self.sym_get(path), path=path) for path in paths]

    def replace_static_text(self, css_selector: str, new_text: str):
        me = pg.clone(self)
        match = me.find_first(css_selector)
        if match is None:
            raise ValueError(f"No match found for {css_selector}")
        if not hasattr(match.node, "set_static_text"):
            raise ValueError(f"Component {match.node} has no set_static_text function.")
        match.node.set_static_text(new_text)
        return me

    @property
    def text(self):
        return None

    async def __call__(
        self, runner: Runner, context: dict, dynamic_context: frozendict | None = None
    ) -> list[LLMResult] | LLMResult:
        key = (id(self), dynamic_context)
        if key not in context:
            context[key] = await self._call(runner, context, dynamic_context)
        return context[key]

    def to_short_string(self, max_depth=None, include_root=True):
        out = list()

        def t(k, v, p):
            if k == "api_config":
                return pg.TraverseAction.CONTINUE
            if not k.is_root:
                if not isinstance(v, (int, float, str, bool)):
                    name = v.__class__.__name__
                else:
                    name = v

                out.append(f"{(k.depth-1)*'   '}- {k.key}: {name}")
            elif include_root:
                out.append(f"{v.__class__.__name__}()")
            if max_depth is not None and k.depth >= max_depth:
                return pg.TraverseAction.CONTINUE
            else:
                return pg.TraverseAction.ENTER

        pg.traverse(self, t)
        return "\n".join(out)

    def plot_program(self, backend="auto"):
        queue = [self]
        nodes = list()
        edges = list()

        def to_list(x):
            if isinstance(x, list):
                return x
            elif isinstance(x, dict):
                return list(x.values())
            else:
                return [x]

        while queue:
            node = queue.pop(0)
            node_data = {
                "id": id(node),
                "label": node.__class__.__name__,
                "priority": len(nodes),
                "details": {
                    "Parameters": (
                        node.to_short_string(max_depth=1, include_root=False)
                        if isinstance(node, Component)
                        else str(node)
                    ),
                },
            }
            nodes.append({"data": node_data})
            children = list()
            if isinstance(node, Symbolic.ObjectType):
                children = node.sym_values()
            for child in children:
                for grandchild in to_list(child):
                    if isinstance(grandchild, Symbolic.ObjectType):
                        edges.append({"data": {"target": id(node), "source": id(grandchild)}})
                        queue.append(grandchild)

        graph = json.dumps(
            {"nodes": nodes, "edges": edges, "node-color": "white", "node-border": 1}, ensure_ascii=False
        )
        html = GRAPH_TEMPLATE.replace("ELEMENTS", graph)
        return HtmlRenderer(html).render(backend)

    def rebind(self, *args, **kwargs):
        """Slightly modified version of the original pyGlove rebind method that allows for deleting values."""
        with pg.allow_partial(True):
            return super().rebind(*args, **kwargs)

    @abc.abstractmethod
    async def _call(
        self, runner: Runner, context: dict, dynamic_context: frozendict | None = None
    ) -> list[LLMResult] | LLMResult:
        pass

    def store_as(self, name: str):
        return StoreAs(self, name)

    @staticmethod
    def _flatten(obj: list):
        results = list()
        for x in obj:
            if not isinstance(x, list):
                results.append(x)
            else:
                results.extend(x)
        flattened = list()
        for r in results:
            flattened += r.values_as_list() if isinstance(r, Result) else [r]
        return flattened


class StoreAs:
    async def _call(
        self, runner: Runner, context: dict, dynamic_context: frozendict | None = None
    ) -> list[LLMResult] | LLMResult:
        result = await self._child(runner, context, dynamic_context)
        if isinstance(result, list):
            return [r.clone_with_stored_value(self._name, r.value) for r in result]
        else:
            return result.clone_with_stored_value(self._name, result.value)


class Template(Component):
    """Simple template-based text component that uses Python's string formatting to fill in
    values. The template variables available are:

    * ``{inputs[id].attribute}`` to refer to a row value
    * ``{constants.attribute}`` to refer to one of the constants

    """

    def __init__(
        self,
        content: str,
        reference_id: str | None = None,
        reference_classes: list[str] | None = None,
        **dependencies: dict,
    ):
        super().__init__(content, reference_id, reference_classes)
        self._children = dependencies
        self.dependencies = [d for d in dependencies.values() if isinstance(d, Component)]
        self._template = self._compile(content)

    @staticmethod
    def _image(this, options=None):
        if options is None:
            return this.get("image", "")
        else:
            return f"{{{{image {options}}}}}"

    @classmethod
    def _compile(cls, template_text: str):
        return pybars.Compiler().compile(template_text)

    async def _call(self, runner: Runner, context: dict | None, dynamic_context: frozendict | None) -> TextResult:
        data = context.get("data", {})
        fill_values = {
            k: await child(runner, context, dynamic_context) if isinstance(child, Component) else child
            for k, child in self._children.items()
        }
        dynamic_context = dict() if dynamic_context is None else dynamic_context
        result = self._fill(**data, **fill_values, **dynamic_context)
        return TextResult(result, parent=list(fill_values.values()) if fill_values else None, op=self)

    def _fill(self, **kwargs) -> str:
        kwargs = {k: self._unwrap_results(v) for k, v in kwargs.items()}
        if len(kwargs.get("inputs", list())) == 1:
            kwargs["input"] = kwargs["inputs"][0]
        return self._template(kwargs, helpers={"image": self._image})

    @property
    def text(self):
        return self._child


class VerbatimText(Template):
    def __init__(self, content: str, reference_id: str | None = None, reference_classes: list[str] | None = None):
        super().__init__(content, reference_id, reference_classes)

    @staticmethod
    def _compile(template_text: str):
        return template_text

    def _fill(self, **kwargs) -> str:
        return self._child
