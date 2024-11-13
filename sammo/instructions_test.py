# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import pytest
from frozendict import frozendict
from sammo.dataformatters import PlainFormatter

from sammo.data import DataTable
from sammo.instructions import MetaPrompt, Section, Paragraph, InputData
from sammo.runners import MockedRunner


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "render_as,expected",
    [
        (
            "xml",
            (
                "<section>\n"
                "<title>Section</title>\n"
                "<section>\n"
                "<title>Subsection</title>Subsection text.\n"
                "</section>\n"
                "</section>"
            ),
        ),
        ("markdown", "# Section\n## Subsection\nSubsection text."),
        ("markdown-alt", "Section\n=======\nSubsection\n----------\nSubsection text."),
    ],
)
async def test_basic_render(render_as, expected):
    runner = MockedRunner()
    rendered = await MetaPrompt(
        [Section(title="Section", content=[Section(title="Subsection", content="Subsection text.")])],
        render_as=render_as,
    )(runner, dict())
    assert rendered.value == expected


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "render_as,expected",
    [
        ("markdown", "# A\nSome text.\n\n\n# B\nOther text."),
    ],
)
async def test_basic_render_text(render_as, expected):
    runner = MockedRunner()
    rendered = await MetaPrompt(
        [Section(title="A", content="Some text.\n"), Section(title="B", content="Other text.")], render_as=render_as
    )(runner, dict())
    assert rendered.value == expected


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "render_as,expected",
    [
        (
            "xml",
            (
                (
                    "<section>\n"
                    "<title>Section</title>\n"
                    "<paragraph>Paragraph 1\n"
                    "</paragraph>\n"
                    "\n"
                    "<paragraph>Paragraph 2\n"
                    "</paragraph>\n"
                    "</section>"
                )
            ),
        ),
        ("markdown", "# Section\nParagraph 1\n\n\nParagraph 2"),
        ("markdown-alt", "Section\n=======\nParagraph 1\n\n\nParagraph 2"),
    ],
)
async def test_paragraph(render_as, expected):
    runner = MockedRunner()
    rendered = await MetaPrompt(
        [Section(title="Section", content=[Paragraph("Paragraph 1"), Paragraph("Paragraph 2")])],
        render_as=render_as,
    )(runner, dict())
    assert rendered.value == expected


@pytest.mark.asyncio
async def test_raw_render():
    runner = MockedRunner()
    rendered = await MetaPrompt(
        [Section("My title\n", "Section 1"), Paragraph("Paragraph 1\n"), Paragraph("Paragraph 2")], render_as="raw"
    )(runner, dict())
    assert rendered.value == "My title\nSection 1Paragraph 1\nParagraph 2"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "render_as,expected",
    [
        (
            "xml",
            (
                "<paragraph>Paragraph 1\n"
                "</paragraph>\n"
                "\n"
                "<paragraph>Input: {'name': 'Alice', 'age': 25}\n"
                "\n"
                "Input: {'name': 'Bob', 'age': 30}\n"
                "\n"
                "</paragraph>"
            ),
        ),
        (
            "markdown",
            (
                "Paragraph 1\n"
                "\n"
                "\n"
                "Input: {'name': 'Alice', 'age': 25}\n"
                "\n"
                "Input: {'name': 'Bob', 'age': 30}"
            ),
        ),
        ("raw", ("Paragraph 1Input: {'name': 'Alice', 'age': 25}\n" "\n" "Input: {'name': 'Bob', 'age': 30}")),
    ],
)
async def test_data_render(render_as, expected):
    data = DataTable([{"name": "Alice", "age": 25}, {"name": "Bob", "age": 30}], [1, 0])
    context = dict(
        data=frozendict(
            inputs=data.inputs.values,
            constants=data.constants,
        )
    )
    data_formatter = PlainFormatter(all_labels=data.outputs.unique(), orient="item")
    runner = MockedRunner()
    rendered = await MetaPrompt(
        [Paragraph("Paragraph 1"), Paragraph(InputData())], render_as=render_as, data_formatter=data_formatter
    )(runner, context)
    assert rendered.value == expected
