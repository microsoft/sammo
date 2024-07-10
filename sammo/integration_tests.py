import pytest

from sammo.base import VerbatimText, Template
from sammo.components import Output, Union, ForEach
from sammo.extractors import ExtractRegex, LambdaExtractor, SplitLines, StripWhitespace
from sammo.runners import MockedRunner


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
