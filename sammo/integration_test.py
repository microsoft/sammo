from io import BytesIO
from unittest.mock import patch

import dill
import pytest

from sammo.base import VerbatimText, Template, EvaluationScore
from sammo.components import Output, Union, ForEach, GenerateText
from sammo.data import DataTable
from sammo.extractors import ExtractRegex, LambdaExtractor, SplitLines, StripWhitespace
from sammo.runners import MockedRunner
from sammo.search import EnumerativeSearch
from sammo.search_op import one_of


def test_manual_loop():
    numbers = [VerbatimText(f"{i}") for i in range(5)]
    result = Output(Union(*numbers)).run(MockedRunner())
    assert result.outputs.values[0] == ["0", "1", "2", "3", "4"]


def test_dynamic_loop():
    numbers = ExtractRegex(
        VerbatimText("<item>1</item><item>2</item><item>3</item>"),
        r"<item>(.*?)<.?item>",
    )
    fruit_blurbs = ForEach(
        "number",
        numbers,
        Template("{{number}}!"),
    )
    result = Output(fruit_blurbs).run(MockedRunner())
    assert result.outputs.values[0] == ["1!", "2!", "3!"]


def test_custom_extractor():
    numbers = [VerbatimText(f"{i}") for i in range(5)]
    numbers = LambdaExtractor(Union(*numbers), "lambda x: int(x) + 1")
    result = Output(numbers).run(MockedRunner())
    assert result.outputs.values[0] == [1, 2, 3, 4, 5]


def test_minibatching():
    data = list(range(5))
    result = Output(
        SplitLines(StripWhitespace(Template("{{#each inputs}}{{this}}\n\n{{/each}}"))),
        minibatch_size=2,
    ).run(MockedRunner(), data, progress_callback=False)
    assert result.outputs.values == [str(d) for d in data]


def test_search():
    def prompt_space():
        prompt = GenerateText(
            Template(f"{{input}}"),
            randomness=one_of([0.3, 0.7, 1.0], reference_id="randomness"),
        )
        return Output(prompt)

    def metric(y_true, y_pred):
        return EvaluationScore(0)

    train_data = DataTable([{"input": "1"}, {"input": "2"}])
    runner = MockedRunner()
    searcher = EnumerativeSearch(runner, prompt_space, metric)
    searcher.fit_transform(train_data)
    with patch("builtins.open", lambda x, y: BytesIO()) as mock_file:
        searcher.save("file.pkl")
