# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import abc
import copy
import re

from beartype.typing import Callable, Self
from frozendict import frozendict
import pyglove as pg
import pybars
from tabulate import tabulate

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
    __slots__ = "parent", "value", "stored_values"

    def __init__(self, value, parent=None, stored_values=None):
        self.value = value
        self.parent = parent
        self.stored_values = stored_values

    def to_json(self):
        return self.value

    @classmethod
    def bfs(cls, start, match_condition: Callable):
        """Breadth-first search returning all nodes that match the given condition.

        Args:
            match_condition: A function that returns True if a node matches."""
        queue = [cls(None, parent=start)]
        matches = list()
        while queue:
            node = queue.pop(0)
            if match_condition(node):
                matches.append(node)
            if isinstance(node, Result):
                queue.extend(node.parents)
        return matches

    def with_parent(self, parent):
        self.parent = parent
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
    def __init__(self, value=None, parent=None, stored_values=None):
        super().__init__(value, parent, stored_values)


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


class CompiledQuery:
    __slots__ = "query", "child_selector"

    def __init__(self, query, child_selector=None):
        self.query = query
        self.child_selector = child_selector

    @classmethod
    def from_path(cls, path_descriptor: str | dict | Self):
        if isinstance(path_descriptor, CompiledQuery):
            return path_descriptor
        elif isinstance(path_descriptor, str):
            path_descriptor = {"regex": path_descriptor}
        child_selector = path_descriptor.pop("_child", None)
        if path_descriptor.get("regex", None) is not None:
            regex = path_descriptor["regex"]
            try:
                re.compile(regex)
            except re.error:
                raise ValueError(f"Invalid regex: {regex}")
            return cls({"path_regex": regex}, child_selector)
        else:
            attribute, value = list(path_descriptor.items())[0]
            if attribute.lower() == "type":
                selector = lambda k, v: isinstance(v, value)
            else:
                selector = (
                    lambda k, v: hasattr(v, "sym_hasattr")
                    and v.sym_hasattr(attribute)
                    and v.sym_get(attribute) == value
                )
            return cls({"custom_selector": selector}, child_selector)

    def __repr__(self):
        return f"CompiledQuery({self.query}, child_selector={self.child_selector})"


@pg.symbolize(eq=True)
class Component:
    """Base class for all components.

    Components are the building blocks of a search space.

    Args:
        child: Child component. This can be another component or a string.
        name: The name of the component. If not provided, the class name is used.
    """

    NEEDS_SCHEDULING = False

    def __init__(self, child: Self | str, name: str | None = None):
        if name is None:
            self._name = self.__class__.__name__
        else:
            self._name = name

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

    def query(self, regex_or_query=None, return_path=False, max_matches=1):
        """Convinience method to query the component tree for a specific component. Uses `pg.query` under the hood.

        :param regex_or_query: A regex string or a query dict or a CompiledQuery object.
        :param return_path: Whether to return a tuple of (path, value) or just the value.
        :param max_matches: The maximum number of matches to return.
        :return: Either component, tuple of (path, component) or None if no match was found.
        """
        compiled_query = CompiledQuery.from_path(regex_or_query)
        matches = list(pg.query(self, **compiled_query.query).items())
        if max_matches:
            matches = matches[:max_matches]

        if compiled_query.child_selector is not None:
            for i in range(len(matches)):
                path, val = matches[i]
                matches[i] = (path + "." + compiled_query.child_selector, val.sym_get(compiled_query.child_selector))

        if not matches:
            return None
        elif return_path:
            return matches[0]
        else:
            return matches[0][1]

    def replace_static_text(self, regex_or_query: str | dict | CompiledQuery, new_text: str):
        me = pg.clone(self)
        result = me.query(regex_or_query, return_path=True)
        if result is None:
            raise ValueError(f"No match found for {regex_or_query}")
        path, component = result
        if not hasattr(component, "set_static_text"):
            raise ValueError(f"Component {component} has no set_static_text function.")
        component.set_static_text(new_text)
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

    def print_structure(self):
        def t(k, v, p):
            if not k.is_root:
                print(f"{k.depth*'   '}{k.key}: {v.__class__.__name__}")
            else:
                print(k, v.__class__.__name__)
            return pg.TraverseAction.ENTER

        pg.traverse(self, t)

    @abc.abstractmethod
    async def _call(
        self, runner: Runner, context: dict, dynamic_context: frozendict | None = None
    ) -> list[LLMResult] | LLMResult:
        pass

    def store_as(self, name: str):
        return StoreAs(self, name)


class ScalarComponent(Component):
    @abc.abstractmethod
    async def _call(self, runner: Runner, context: dict, dynamic_context: frozendict | None = None) -> LLMResult:
        pass


class ListComponent(Component):
    @abc.abstractmethod
    async def _call(self, runner: Runner, context: dict, dynamic_context: frozendict | None = None) -> list[LLMResult]:
        pass

    @staticmethod
    def _flatten(obj: list):
        results = list()
        for x in obj:
            if not isinstance(x, list):
                results.append(x)
            else:
                results.extend(x)

        return results


class StoreAs:
    async def _call(
        self, runner: Runner, context: dict, dynamic_context: frozendict | None = None
    ) -> list[LLMResult] | LLMResult:
        result = await self._child(runner, context, dynamic_context)
        if isinstance(result, list):
            return [r.clone_with_stored_value(self._name, r.value) for r in result]
        else:
            return result.clone_with_stored_value(self._name, result.value)


class Template(ScalarComponent):
    """Simple template-based text component that uses Python's string formatting to fill in
    values. The template variables available are:

    * ``{inputs[id].attribute}`` to refer to a row value
    * ``{constants.attribute}`` to refer to one of the constants

    """

    def __init__(
        self,
        template_text: str,
        name: str | None = None,
        **dependencies: dict,
    ):
        super().__init__(template_text, name)
        self._children = dependencies
        self.dependencies = [d for d in dependencies.values() if isinstance(d, Component)]
        self._template = self._compile(template_text)

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
        return TextResult(result)

    def _fill(self, **kwargs) -> str:
        kwargs = {k: self._unwrap_results(v) for k, v in kwargs.items()}
        if len(kwargs.get("inputs", list())) == 1:
            kwargs["input"] = kwargs["inputs"][0]
        return self._template(kwargs, helpers={"image": self._image})

    @property
    def text(self):
        return self._child


class VerbatimText(Template):
    def __init__(self, template: str, name: str | None = None):
        super().__init__(template, name)

    @staticmethod
    def _compile(template_text: str):
        return template_text

    def _fill(self, **kwargs) -> str:
        return self._child
