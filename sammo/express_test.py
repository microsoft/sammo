import textwrap
from sammo.express import ExpressParser, _extract_html_comment, _get_ids_and_classes

COMPLEX = """
# Heading 1  <!-- .header -->
A long trip.

* list **item** 1 <!-- #id1 #id2 .class1 .class2 -->
* list item 2

## Heading 1.2
A short trip

## Heading 1.3

# Heading 2
Another long trip.

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

## Heading 2.1
And so **on**.

# Heading 3
[ref](https://www.google.com)
"""


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


def test_express_parser_parse_annotated_markdown():
    input_text = textwrap.dedent(
        """
    # Heading 1
    Some content
    * list item 1 <!-- #id1 .class1 -->
    * list item 2
    """
    )
    parser = ExpressParser(input_text)
    assert parser.parsed_tree is not None
    assert parser.parsed_config == {}


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
    parser = ExpressParser(input_text)
    sammo_tree = parser._aux_tree_to_sammo(parser.parsed_tree)
    assert sammo_tree is not None


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
    parser = ExpressParser(input_text)
    assert parser.parsed_config == {
        "mutators": [{"name": "mutator1", "type": "type1"}, {"name": "mutator2", "type": "type2"}]
    }
