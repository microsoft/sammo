"""
This module contains a variety of search operators that can be used to define a discrete search space for `GridSearch`
or define a set of initial candidates for other search algorithms.
"""
from beartype.typing import Iterable, Any
from pyglove.core.hyper import OneOf, ManyOf

__all__ = ["one_of", "many_of", "permutate", "optional"]


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
