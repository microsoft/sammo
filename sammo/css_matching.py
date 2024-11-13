# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import pyglove as pg
from lxml import etree
from lxml.cssselect import CSSSelector


class XmlTree:
    def __init__(self, root, sym_path=None):
        self.root = root
        self.sym_path = sym_path

    @classmethod
    def from_pyglove(cls, pg_node, treat_as_attributes=({"reference_id": "id", "reference_classes": "class"})):
        if pg.is_abstract(pg_node):
            return ValueError("PyGlove object needs to be fully instantiated.")

        root = etree.Element("root")
        sym_path = {root: ""}

        # Do a breadth-first search
        fifo_queue = [(pg_node, root)]
        while fifo_queue:
            node, parent = fifo_queue.pop(0)

            if isinstance(node, list):
                for i, v in enumerate(node):
                    xml_node = etree.SubElement(parent, v.__class__.__name__.lower())
                    sym_path[xml_node] = v.sym_path if hasattr(v, "sym_path") else node.sym_path + f"[{i}]"
                    fifo_queue.append((v, xml_node))
            elif isinstance(node, (pg.Object, dict)):
                for k, v in node.sym_items() if isinstance(node, pg.Object) else node.items():
                    if k in treat_as_attributes:
                        if v is not None:
                            parent.attrib[treat_as_attributes[k]] = " ".join(v) if isinstance(v, list) else str(v)
                    else:
                        xml_node = etree.SubElement(parent, k)
                        sym_path[xml_node] = v.sym_path if hasattr(v, "sym_path") else node.sym_path + f".{k}"

                        if v is not None and not isinstance(v, (str, int, list, float, pg.List)):
                            xml_node = etree.SubElement(xml_node, v.__class__.__name__.lower())

                        sym_path[xml_node] = v.sym_path if hasattr(v, "sym_path") else node.sym_path + f"[{k}]"
                        fifo_queue.append((v, xml_node))

            elif isinstance(node, (str, int, float)):
                parent.text = str(node)
            elif node is not None:
                raise ValueError(f"Unsupported type: {type(node)}")
        return cls(root, sym_path)

    @staticmethod
    def to_string(xml_node) -> str:
        if xml_node is not None:
            return etree.tostring(xml_node, pretty_print=True).decode()
        else:
            return "None"

    def __repr__(self):
        return self.to_string(self.root)

    def __str__(self):
        return self.to_string(self.root)

    def find_all(self, css_expression: str) -> list:
        selector = CSSSelector(css_expression)
        matches = [match for match in selector(self.root)]
        return [self.sym_path[match] for match in matches]
