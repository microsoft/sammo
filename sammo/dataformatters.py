# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
DataFormatters are components that take a DataTable or dict and format it into a string. They also provide a
get_extractor method that can be used to parse the LLM responses in this format.

"""
import collections
import json

from beartype.typing import Sequence, Literal
from frozendict import frozendict
import xmltodict

from sammo.base import Component, Runner
from sammo.data import OutputAccessor
from sammo.extractors import JSONPath, ParseJSON, ExtractRegex, StripWhitespace, ParseXML

__all__ = ["DataFormatter", "JSONDataFormatter", "XMLDataFormatter", "MultiLabelFormatter", "QuestionAnswerFormatter"]


class LongFormatData:
    """Intermediate data representation that allows for easy grouping of records by kind."""

    __slots__ = ["_records", "_default_sort"]

    def __init__(self, records: list[dict], default_sort=("id", "kind_order")):
        self._records = records
        self._default_sort = default_sort

    @property
    def records(self):
        return self._records

    def _group_by(
        self, fields: Sequence[str] | str, sort_by: Sequence[str] | str | None = None
    ) -> list[tuple[dict, list[dict]]] | list[tuple[str, list[dict]]]:
        data = self._records
        if sort_by is not None:
            sort_by = [sort_by] if isinstance(sort_by, str) else sort_by
            data = sorted(data, key=lambda x: tuple(x[s] for s in sort_by))

        fields = fields if isinstance(fields, list) else [fields]
        grouped = collections.OrderedDict()

        for item in data:
            key = tuple((f, item[f]) for f in fields)
            if key not in grouped:
                grouped[key] = list()
            grouped[key].append({k: v for k, v in item.items() if k not in fields})
        return [(dict(k) if len(k) > 1 else k[0][1], v) for k, v in grouped.items()]

    def group_by(self, orient: str) -> list[tuple[dict, list[dict]]] | list[tuple[str, list[dict]]]:
        if orient == "kind":
            return self._group_by(["kind_alias", "kind_order", "kind"], self._default_sort)
        else:
            return self._group_by("id", self._default_sort)


class DataFormatter(Component):
    """
    A DataFormatter is a component that takes a DataTable or dict and formats it into a string.

    :param names: A dictionary mapping column (e.g., "labels") to a name (e.g., "Input").
    :param flatten_1d_dicts: If True, dicts with a single key are unwrapped to their value.
    :param include_ids: If True, numerical item ids are included in the output.
    :param orient: If "item_id", output format is a series of item records, each with an id and a value. If "kind",
        all inputs or output labels are grouped together.
    :param all_labels: A list of all possible labels, used by some formatters to determine the extractor.
    """

    DEFAULT_NAMES = {"input": "input", "gold_label": "output", "predicted_label": "predicted_output"}

    def __init__(
        self,
        names: dict | None = None,
        flatten_1d_dicts: bool = True,
        include_ids: bool = True,
        orient: Literal["item", "kind"] = "item",
        all_labels=None,
        attributes_processor=None,
    ):
        super().__init__(None)
        self._format = format
        self._names = self.DEFAULT_NAMES
        if names is not None:
            self._names = {**self._names, **names}
        self._include_ids = include_ids
        self._flatten_1d_dicts = flatten_1d_dicts
        self._attributes_processor = attributes_processor
        self._orient = "id" if orient.startswith("item") else "kind"

    def format_datatable(self, data, offset: int = 0):
        return self.format_batch(
            data.inputs.values,
            data.outputs.values,
            offset=offset,
        )

    def format_single(
        self,
        attributes: dict = None,
        gold_label: dict = None,
        predicted_label: dict = None,
        x_id: int = 0,
    ) -> str:
        return self.format_batch([attributes], [gold_label], [predicted_label], x_id)

    def format_batch(
        self,
        attributes: list[dict],
        gold_label: list[dict] = None,
        predicted_label: list[dict] = None,
        offset: int = 0,
    ):
        records = list()
        batch = {"input": attributes, "gold_label": gold_label, "predicted_label": predicted_label}
        batch = {k: v for k, v in batch.items() if v is not None and v != [None]}
        for kind_order, (kind, values) in enumerate(batch.items()):
            for i, v in enumerate(values):
                if isinstance(v, dict) and len(v) == 1 and self._flatten_1d_dicts:
                    v = list(v.values())[0]
                if self._attributes_processor is not None:
                    v = self._attributes_processor(v)
                records.append(
                    {
                        "id": offset + i,
                        "kind": kind,
                        "kind_alias": self._names[kind],
                        "kind_order": kind_order,
                        "value": v,
                    }
                )
        return self._dump(LongFormatData(records))

    async def __call__(
        self,
        runner: Runner,
        data: frozendict | None,
        seed: int | None,
    ) -> list[dict]:
        return [{"value": self.format_batch(data), "name": self._name}]

    def _dump(self, records: list[dict]) -> str:
        pass


class JSONDataFormatter(DataFormatter):
    def __init__(self, newline_delimited=False, indent=None, **kwargs):
        super().__init__(**kwargs)
        self._newline_delimited = newline_delimited
        self._json_params = dict(sort_keys=False, ensure_ascii=False, indent=indent)

    def _dump(self, records: list[dict]):
        grouped = records.group_by(self._orient)

        if self._orient == "id":
            finalized = list()
            for item_id, item_info in grouped:
                record = dict()
                if self._include_ids:
                    record["id"] = item_id
                for info in item_info:
                    record[info["kind_alias"]] = info["value"]
                finalized.append(record)
        elif self._orient == "kind":
            finalized = dict()
            for kind, items in grouped:
                finalized[kind["kind_alias"]] = list()
                for item in items:
                    if self._include_ids:
                        record = {"id": item["id"], "value": item["value"]}
                    else:
                        record = item["value"]
                    finalized[kind["kind_alias"]].append(record)

        if self._newline_delimited and self._orient == "id":
            return "\n".join(json.dumps(x, **self._json_params) for x in finalized)
        elif self._newline_delimited and self._orient == "kind":
            return "\n".join(f"{k}: {json.dumps(v, **self._json_params)}" for k, v in finalized.items())
        else:
            return json.dumps(finalized, **self._json_params)

    def get_extractor(self, child, on_error="raise"):
        return JSONPath(
            ParseJSON(child, parse_fragments="all", on_error=on_error),
            f"$..{self._names['gold_label'] if self._orient == 'id' else 'value'}",
            flatten_lists=False,
        )


class MultiLabelFormatter(DataFormatter):
    def __init__(self, all_labels: list, **kwargs):
        super().__init__(flatten_1d_dicts=True, include_ids=False, **kwargs)
        unwrapped = [OutputAccessor.unwrap(l) for l in all_labels]
        labels_are_scalars = all([isinstance(x, (int, str, float)) for x in unwrapped])
        self._labels = unwrapped
        if not labels_are_scalars:
            raise ValueError("MultiLabelFormatter can only be used with scalar labels.")

    def _dump(self, records: list[dict]):
        grouped = records.group_by("kind")
        result = list()
        for group, items in grouped:
            kind = group["kind"]
            if kind == "input":
                result.append(group["kind_alias"] + ": " + json.dumps([item["value"] for item in items]))
            else:
                result.append(group["kind_alias"] + ": " + " ".join([item["value"] for item in items]))

        return "\n".join(result)

    def get_extractor(self, child, on_error="raise"):
        return ExtractRegex(child, "|".join(self._labels))


class QuestionAnswerFormatter(MultiLabelFormatter):
    def _dump(self, records: list[dict]):
        grouped = records.group_by(self._orient)
        result = list()
        if self._orient == "id":
            for i, (item_id, item_infos) in enumerate(grouped):
                for item in item_infos:
                    prefix = "Q" if item["kind"] == "input" else "A"
                    result += [f"{prefix}[{i}]: {item['value']}"]
                result += [""]
        else:
            for group, items in grouped:
                prefix = "Q" if group["kind"] == "input" else "A"
                result += [f"{prefix}[{i}]: {item['value']}" for i, item in enumerate(items)]

        return "\n".join(result)

    def get_extractor(self, child, on_error="raise"):
        return ExtractRegex(child, r"^\s*A[^:]*:\s*([^\n]*)")


class PlainFormatter(DataFormatter):
    MAP = {"input": "Input", "gold_label": "Output", "predicted_label": "predicted_label"}

    def __init__(self, **kwargs):
        super().__init__(flatten_1d_dicts=True, include_ids=False, **kwargs)

    def _dump(self, records: list[dict]):
        grouped = records.group_by(self._orient)
        result = list()
        if self._orient == "id":
            for i, (item_id, item_infos) in enumerate(grouped):
                for item in item_infos:
                    result += [f"{self.MAP[item['kind']]}: {item['value']}"]
                result += [""]
        else:
            for group, items in grouped:
                result += [f"{self.MAP[group['kind']]}: {item['value']}" for i, item in enumerate(items)]

        return "\n".join(result)

    def get_extractor(self, child, on_error="raise"):
        return StripWhitespace(child)


class XMLDataFormatter(DataFormatter):
    def _render_record(self, item_id: int, value, convert_lists=False):
        if self._include_ids:
            return {"@id": item_id, "value": value}
        elif isinstance(value, (list, tuple)) and convert_lists:
            return [{"value": v} for v in value]
        else:
            return value

    def _dump(self, records: list[dict]):
        grouped = records.group_by(self._orient)
        finalized = list()
        if self._orient == "id":
            for item_id, item_info in grouped:
                record = dict()
                for info in item_info:
                    record[info["kind_alias"]] = self._render_record(item_id, info["value"])
                finalized.append(record)
        elif self._orient == "kind":
            for kind, items in grouped:
                for item in items:
                    record = self._render_record(item["id"], item["value"], convert_lists=True)
                    finalized.append({kind["kind_alias"]: record})
        return "\n".join(xmltodict.unparse(x, full_document=False, pretty=True) for x in finalized)

    def get_extractor(self, child, on_error="raise"):
        return JSONPath(
            ParseXML(child, parse_fragments="all", on_error=on_error),
            f'$..{self._names["gold_label"]}',
            flatten_lists=True,
        )
