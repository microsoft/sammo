import textwrap
from sammo.express import MarkdownParser, _extract_html_comment, _get_ids_and_classes
import pyglove as pg


def test_extract_html_comment():
    text = "Some text <!-- This is a comment -->more text"
    comment, rest = _extract_html_comment(text)
    assert comment == " This is a comment "
    assert rest == "Some text more text"


def test_extract_html_comment_no_comment():
    text = "Some text more text"
    comment, rest = _extract_html_comment(text)
    assert comment == ""
    assert rest == text


def test_get_ids_and_classes():
    text = "Some text <!-- #id1 .class1 #id2 .class2 --> more text"
    result = _get_ids_and_classes(text)
    assert result["text"] == "Some text  more text"
    assert set(result["ids"]) == {"id1", "id2"}
    assert set(result["classes"]) == {"class1", "class2"}


def test_get_ids_and_classes_no_comment():
    text = "Some text more text"
    result = _get_ids_and_classes(text)
    assert result["text"] == text
    assert result["ids"] == []
    assert result["classes"] == []


def test_lists():
    input_text = textwrap.dedent(
        """
    * list item 1
    * list item 2
    """
    )

    expected = (
        {
            "children": [{"children": ["* list item 1\n", "* list item 2\n"], "class": [], "id": [], "type": "list"}],
            "type": "root",
        },
        {},
    )
    parsed = MarkdownParser._parse_annotated_markdown(input_text)
    assert parsed == expected


def test_nested_sections():
    input_text = textwrap.dedent(
        """
    # Heading 1
    Some content
    ## Heading 1.1
    More content
    ### Heading 1.1.1
    Even more content
    # Heading 2
    Final content
    """
    )
    expected = pg.from_json(
        {
            "_type": "sammo.instructions.MetaPrompt",
            "child": [
                {
                    "_type": "sammo.instructions.Section",
                    "title": "# Heading 1\n",
                    "content": [
                        {
                            "_type": "sammo.instructions.Paragraph",
                            "content": ["Some content\n"],
                            "reference_id": None,
                            "reference_classes": None,
                        },
                        {
                            "_type": "sammo.instructions.Section",
                            "title": "## Heading 1.1\n",
                            "content": [
                                {
                                    "_type": "sammo.instructions.Paragraph",
                                    "content": ["More content\n"],
                                    "reference_id": None,
                                    "reference_classes": None,
                                },
                                {
                                    "_type": "sammo.instructions.Section",
                                    "title": "### Heading 1.1.1\n",
                                    "content": [
                                        {
                                            "_type": "sammo.instructions.Paragraph",
                                            "content": ["Even more content\n"],
                                            "reference_id": None,
                                            "reference_classes": None,
                                        }
                                    ],
                                    "reference_id": None,
                                    "reference_classes": None,
                                },
                            ],
                            "reference_id": None,
                            "reference_classes": None,
                        },
                    ],
                    "reference_id": None,
                    "reference_classes": None,
                },
                {
                    "_type": "sammo.instructions.Section",
                    "title": "# Heading 2\n",
                    "content": [
                        {
                            "_type": "sammo.instructions.Paragraph",
                            "content": ["Final content\n"],
                            "reference_id": None,
                            "reference_classes": None,
                        }
                    ],
                    "reference_id": None,
                    "reference_classes": None,
                },
            ],
            "render_as": "raw",
            "data_formatter": None,
            "reference_id": None,
            "seed": 0,
        }
    )
    parser = MarkdownParser(input_text)
    assert parser.get_sammo_program() == expected


def test_express_parser_parse_annotated_markdown():
    input_text = textwrap.dedent(
        """
    # Heading 1
    Some content
    * list item 1 <!-- #id1 .class1 -->
    * list item 2

    ## Heading 1.2 <!-- #id2 .class2 .class3 -->
    {{{input}}}
    """
    )
    parser = MarkdownParser(input_text)
    expected = pg.from_json(
        {
            "_type": "sammo.instructions.MetaPrompt",
            "child": [
                {
                    "_type": "sammo.instructions.Section",
                    "title": "# Heading 1\n",
                    "content": [
                        {
                            "_type": "sammo.instructions.Paragraph",
                            "content": ["Some content\n"],
                            "reference_id": None,
                            "reference_classes": None,
                        },
                        {
                            "_type": "sammo.instructions.Paragraph",
                            "content": ["* list item 1 \n", "* list item 2\n"],
                            "reference_id": "id1",
                            "reference_classes": ["class1"],
                        },
                        {
                            "_type": "sammo.instructions.Section",
                            "title": "## Heading 1.2 \n",
                            "content": [
                                {
                                    "_type": "sammo.instructions.Paragraph",
                                    "content": [
                                        {
                                            "_type": "sammo.base.Template",
                                            "content": "{{{input}}}\n",
                                            "reference_id": None,
                                            "reference_classes": None,
                                        }
                                    ],
                                    "reference_id": None,
                                    "reference_classes": None,
                                }
                            ],
                            "reference_id": "id2",
                            "reference_classes": ["class2", "class3"],
                        },
                    ],
                    "reference_id": None,
                    "reference_classes": None,
                }
            ],
            "render_as": "raw",
            "data_formatter": None,
            "reference_id": None,
            "seed": 0,
        }
    )
    assert parser.get_sammo_program() == expected
    assert parser.get_sammo_config() == {}


def test_express_parser_aux_tree_to_sammo():
    input_text = textwrap.dedent(
        """
    # Heading 1
    Some content
    ```{python}
    print("Hello, World!")
    ```
    """
    )
    parser = MarkdownParser(input_text)
    assert parser.get_sammo_program() is not None


def test_express_parser_with_mutators():
    input_text = textwrap.dedent(
        """
    # Heading 1
    Some content
    > Somewhere, something incredible is waiting to be known

    ```{sammo/mutators}
    {
      "mutators": [
        {
          "name": "mutator1",
          "type": "type1"
        },
        {
          "name": "mutator2",
          "type": "type2"
        }
      ]
    }
    ```
    """
    )
    parser = MarkdownParser(input_text)
    assert parser.get_sammo_config() == {
        "mutators": [{"name": "mutator1", "type": "type1"}, {"name": "mutator2", "type": "type2"}]
    }
