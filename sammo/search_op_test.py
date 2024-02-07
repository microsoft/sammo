# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from sammo.search_op import one_of, many_of, permutate, optional, get_points_from_search_space
import pyglove as pg


@pg.symbolize(eq=True)
class Demo:
    def __init__(self, params):
        self.params = params


def enumerate_candidates(search_space):
    traced_search_space = pg.hyper.trace(search_space)
    all_candidates = list()
    for search_context in pg.iter(traced_search_space):
        with search_context():
            all_candidates.append(search_space())
    return all_candidates


def test_one_of():
    space = lambda: one_of([1, 2, 3])
    assert enumerate_candidates(space) == [1, 2, 3]


def test_many_of():
    space = lambda: many_of(2, [1, 2, 3])
    assert enumerate_candidates(space) == [[1, 2], [1, 3], [2, 3]]


def test_permutate():
    space = lambda: permutate([1, 2, 3])
    assert sorted(enumerate_candidates(space)) == sorted(
        [
            [1, 2, 3],
            [1, 3, 2],
            [2, 1, 3],
            [3, 1, 2],
            [2, 3, 1],
            [3, 2, 1],
        ]
    )


def test_optional():
    space = lambda: optional(1)
    assert enumerate_candidates(space) == [[], [1]]


def test_get_points_from_search_space():
    me = Demo(one_of(["a", "b"]))
    points = get_points_from_search_space(me, 2, sample=False)
    assert points == [Demo("a"), Demo("b")]


def test_get_points_from_search_space_with_names():
    me = lambda: Demo(one_of(["a", "b"], name="params"))
    points, names = get_points_from_search_space(me, 2, sample=False, return_names=True)
    assert points == [Demo("a"), Demo("b")]
    assert names == [{"params": "'a'"}, {"params": "'b'"}]
