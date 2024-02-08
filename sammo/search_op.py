# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
This module contains a variety of search operators that can be used to define a discrete search space for `GridSearch`
or define a set of initial candidates for other search algorithms.
"""
import pyglove as pg
from sammo.base import Component
from beartype.typing import Iterable, Any, Callable
from pyglove.core.hyper import OneOf, ManyOf

__all__ = ["one_of", "many_of", "permutate", "optional", "get_points_from_search_space", "get_first_point"]


def get_points_from_search_space(
    search_space: Callable | Component | list | dict,
    n_points: int,
    sample: bool = False,
    seed: int = 42,
    return_names: bool = False,
) -> list[Component]:
    """Materialize a number of points from a search space.

    :param search_space: Search space, either represented as function or a single Output class.
    :param n_points: Number of points to materialize.
    :param sample: Whether to sample from the search space or enumerate and return first `n_points`.
    :param seed: Random seed for sampling.
    :param return_names: Whether to return the names of the points.
    """
    names = list()
    if isinstance(search_space, list):
        search_space = pg.list(search_space)
    elif isinstance(search_space, dict):
        search_space = pg.dict(search_space)
    if isinstance(search_space, Callable):
        candidates = list()
        for context in pg.iter(
            pg.hyper.trace(search_space),
            num_examples=n_points,
            algorithm=pg.geno.Random(seed) if sample else None,
        ):
            names.append(context.__closure__[0].cell_contents.to_dict("name_or_id", "literal"))
            with context():
                candidates.append(search_space())
    elif search_space.is_deterministic:
        candidates = [search_space] * n_points
    elif sample:
        candidates = list(pg.random_sample(search_space, n_points, seed=seed))
    else:
        candidates = list(pg.iter(search_space, n_points))
    if return_names:
        return candidates, names
    else:
        return candidates


def get_first_point(search_space: Callable | Component | list | dict) -> Component:
    """Return the first value of the enumerated search space.

    :param search_space: Search space, either represented as function or a single Output class."""
    return get_points_from_search_space(search_space, 1, sample=False)[0]


class OneOfPatched(OneOf):
    def __getitem__(self, item):
        return ""


class ManyOfPatched(ManyOf):
    def __getitem__(self, item):
        return ""


def one_of(candidates: Iterable, name: str | None = None) -> Any:
    """Search operator for selecting one of the given candidates.

    :param candidates: The list of candidates to choose from.
    :param name: The name of the operator for later reference.
    """
    return OneOfPatched([(lambda n=x: n)(x) if not callable(x) else x for x in candidates], name=name)


def many_of(num_choices: int, candidates: Iterable, name: str | None = None) -> Any:
    """Search operator for n choose k.

    :param num_choices: The number of candidates to choose.
    :param candidates: The list of candidates to choose from.
    :param name: The name of the operator for later reference.
    """
    return ManyOfPatched(
        num_choices=num_choices,
        choices_sorted=True,
        candidates=[(lambda n=x: n)(x) if not callable(x) else x for x in candidates],
        name=name,
    )


def permutate(candidates: Iterable, name: str | None = None) -> Any:
    """Search operator for permutating a list of components.

    :param candidates: The list of components to permute.
    :param name: The name of the operator for later reference.
    """
    return ManyOfPatched(
        num_choices=len(list(candidates)),
        choices_distinct=True,
        choices_sorted=False,
        candidates=[(lambda n=x: n)(x) if not callable(x) else x for x in candidates],
        name=name,
    )


def optional(candidate, name=None) -> Any:
    """Search operator for making a component optional.

    :param val: The value to include or exclude.
    :param name: The name of the operator for later reference.
    """
    return one_of([[], [candidate]])
