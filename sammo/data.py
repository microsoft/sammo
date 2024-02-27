# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
DataTables are the primary data structure used in SAMMO.
They are essentially a wrapper around a list of inputs and outputs (labels), with some additional functionality.
"""
import copy
import hashlib
import math

from beartype import beartype
from beartype.typing import Callable, Iterator, Self
import more_itertools
import orjson
import pyglove as pg
import random
import tabulate

from sammo.utils import serialize_json


# monkey-patch to fix bug in tabulate with booleans and multline
def _force_str(*args, **kwargs):
    return str


tabulate._type = _force_str

from sammo.base import EmptyResult, LLMResult, NonEmptyResult, Result, TimeoutResult, Costs

__all__ = ["DataTable"]


@beartype
class DataTable(pg.JSONConvertible):
    def __init__(
        self,
        inputs: list,
        outputs: list | None = None,
        constants: dict | None = None,
        seed=42,
    ):
        inputs = DataTable._ensure_list(inputs)
        outputs = DataTable._ensure_list(outputs or [None] * len(inputs))

        if len(inputs) != len(outputs):
            raise ValueError(f"Input fields have {len(inputs)} rows, but output fields have {len(outputs)} rows.")

        self._data = {
            "outputs": outputs,
            "inputs": inputs,
            "constants": constants,
        }

        self._inputs = Accessor(self, "inputs")
        self._outputs = OutputAccessor(self, "outputs")
        self._len = len(outputs)
        self._seed = seed
        self._rng = random.Random(seed)
        self._hash = None

    def __len__(self):
        return self._len

    @property
    def inputs(self):
        """Access input data."""
        return self._inputs

    @property
    def outputs(self):
        """Access output data."""
        return self._outputs

    @property
    def constants(self) -> dict | None:
        """Access constants."""
        return self._data["constants"]

    def to_json(self, **kwargs):
        """Convert to a JSON-serializable object.

        .. note::
           This only saves the values of the outputs (shallow state), not the raw results.
        """
        data = self.to_records(only_values=True)
        return {"_type": "DataTable", "records": data, "constants": self.constants, "seed": self._seed}

    def persistent_hash(self):
        if self._hash is None:
            serialized = serialize_json((self.to_records(only_values=False), self.constants, self._seed))
            self._hash = int(hashlib.md5(serialized).hexdigest(), 16)
        return self._hash

    @classmethod
    def from_json(cls, json_value, **kwargs):
        data = json_value["records"]
        return cls.from_records(data, constants=json_value["constants"], seed=json_value["seed"])

    @classmethod
    def from_pandas(
        cls,
        df: "pandas.DataFrame",
        output_fields: list[str] | str = "output",
        input_fields: list[str] | str | None = None,
        constants: dict | None = None,
        seed=42,
    ):
        """Create a DataTable from a pandas DataFrame.

        :param df: Pandas DataFrame.
        :param input_fields: Columns from pandas DataFrame that will be used as inputs.
        :param output_fields: Columns that will be used as outputs or targets (e.g., labels).
        :param constants: Constants.
        :param seed: Random seed.
        """
        return cls.from_records(
            df.to_dict("records"),
            output_fields=output_fields,
            input_fields=input_fields,
            constants=constants,
            seed=seed,
        )

    def _slice_to_explicit_idx(self, key: slice):
        max_idx = len(self)
        return list(range(key.start or 0, min(key.stop or max_idx, max_idx), key.step or 1))

    @classmethod
    def from_records(
        cls,
        records: list[dict],
        output_fields: list[str] | str = "output",
        input_fields: list[str] | str | None = None,
        **kwargs,
    ):
        if len(records) == 0:
            return cls(records, **kwargs)

        if input_fields is None:
            input_fields = [k for k in records[0].keys() if k not in output_fields]
        input_fields = cls._ensure_list(input_fields)
        output_fields = cls._ensure_list(output_fields)

        outputs, inputs = list(), list()
        for r in records:
            r_att = {k: v for k, v in r.items() if k in input_fields}
            r_lab = {k: v for k, v in r.items() if k in output_fields}

            inputs.append(r_att[input_fields[0]] if len(input_fields) == 1 else r_att)
            outputs.append(r_lab[output_fields[0]] if len(output_fields) == 1 else r_lab)

        return cls(inputs, outputs, **kwargs)

    def to_records(self, only_values=True):
        """Convert to a list of dictionaries.

        :param only_values: If False, raw result objects will be returned for `.outputs`."""
        inputs = self.inputs.values
        outputs = self.outputs.values if only_values else self.outputs.raw_values
        return [{"input": a, "output": l} for a, l in zip(inputs, outputs)]

    @staticmethod
    def _ensure_list(val):
        if isinstance(val, list):
            return val
        else:
            return [val]

    @staticmethod
    def _truncate(x, max_length=25):
        x = str(x)
        if len(x) > max_length:
            return x[: max_length - 3] + "..."
        return x

    def __repr__(self):
        return self.to_string()

    def to_string(self, max_rows: int = 10, max_col_width: int = 60, max_cell_length: int = 500):
        """Convert to a printable string.

        :param max_rows: Maximum number of rows to include. Defaults to 10.
        :param max_col_width: Maximum width of each column. Defaults to 50.
        :param max_cell_length: Maximum characters in each cell. Defaults to 100.
        """
        table_data = [
            {str(k): DataTable._truncate(v, max_cell_length) for k, v in x.items()} for x in self.to_records()
        ]
        if table_data:
            table = tabulate.tabulate(
                table_data[:max_rows], headers="keys", maxcolwidths=max_col_width, tablefmt="grid"
            )
        else:
            table = "<empty DataTable>"
        return f"{table}\nConstants: {DataTable._truncate(self.constants, max_col_width)}"

    def _to_explicit_idx(self, key: int | slice | list[int]):
        if isinstance(key, int):
            return [key]
        elif isinstance(key, slice):
            key = self._slice_to_explicit_idx(key)
        return key

    def __getitem__(self, key):
        idx = self._to_explicit_idx(key)
        new_inputs = [self._data["inputs"][i] for i in idx]
        new_outputs = [self._data["outputs"][i] for i in idx]
        return DataTable(new_inputs, new_outputs, self.constants, self._seed)

    def sample(self, k: int, seed: int | None = None) -> Self:
        """Sample rows without replacement.

        :param k: Number of rows to sample.
        :param seed: Random seed. If not provided, instance seed is used.
        """
        if k > len(self):
            raise ValueError("Sample size must be less than or equal to the number of rows.")
        if seed is not None:
            rng = random.Random(seed)
        else:
            rng = self._rng
        selected_idx = rng.sample(range(len(self)), k=k)

        return self[selected_idx]

    def shuffle(self, seed: int | None = None) -> Self:
        """Shuffle rows.

        :param seed: Random seed. If not provided, instance seed is used.
        """
        return self.sample(len(self), seed=seed)

    def random_split(self, *sizes: int, seed=None) -> tuple:
        """
        Randomly split the dataset into non-overlapping new datasets of given lengths.
        :param sizes: Sizes of splits to be produced, sum of sizes may not exceed length of the dataset.
        :param seed: Random seed. If not provided, instance seed is used.
        :return: Tuple of splits.
        """
        sampled = self.sample(sum(sizes), seed=seed)
        splits = [slice(sum(sizes[:i]), sum(sizes[: i + 1])) for i in range(len(sizes))]
        return tuple(sampled[split] for split in splits)

    def copy(self) -> Self:
        return copy.deepcopy(self)

    def get_minibatch_iterator(self, minibatch_size):
        return MinibatchIterator(self, minibatch_size)


DataTable.register("DataTable", DataTable)


class Accessor:
    def __init__(self, parent, group):
        self._parent = parent
        self._group = group

    @staticmethod
    def _safe_get(y, field):
        if isinstance(y, dict) and field in y:
            return y[field]
        elif hasattr(y, field):
            return getattr(y, field)
        else:
            return None

    @property
    def raw_values(self):
        """Return the raw data."""
        return self._parent._data[self._group]

    def __getitem__(self, key):
        if isinstance(key, int):
            return self.raw_values[key]
        elif isinstance(key, slice):
            idx = self._parent._slice_to_explicit_idx(key)
            return [self.raw_values[i] for i in idx]
        elif isinstance(key, list):
            return [self.raw_values[i] for i in key]
        else:
            raise ValueError(f"Invalid key type: {type(key)}")

    def field(self, name: str) -> DataTable:
        """Return a new DataTable with only the given field.

        :param name: Name of the field.
        """
        new_data = [Accessor._safe_get(y, name) for y in self.raw_values]
        clone = copy.deepcopy(self._parent)
        clone._data[self._group] = new_data
        return clone

    def unique(self) -> list:
        """Return unique values of current subset (input or output)."""
        data = self.raw_values
        unique = {orjson.dumps(x): x for x in data}
        return list(unique.values())

    def filtered_on(self, condition: Callable[[dict], bool]) -> DataTable:
        """Filter rows based on a condition.

        :param condition: A function that takes a row and returns a boolean.
        """
        sel_idx = [i for i, x in enumerate(self.raw_values) if condition(x)]
        return self._parent[sel_idx]

    @property
    def values(self):
        return self._get_values(self.raw_values)

    @classmethod
    def _get_values(self, data):
        if isinstance(data, list):
            return [self._get_values(x) for x in data]
        elif isinstance(data, NonEmptyResult):
            return data.value
        else:
            return data


class OutputAccessor(Accessor):
    @staticmethod
    def unwrap(x, on_empty=None, ignore=tuple(), flatten_1d_dicts=True):
        while True:
            if isinstance(x, dict):
                x = {k: v for k, v in x.items() if k not in ignore}
                if flatten_1d_dicts and len(x) == 1:  # and not isinstance(list(x.values())[0], (int, float, str)):
                    x = list(x.values())[0]
                else:
                    return x
            elif isinstance(x, list) and len(x) == 1:
                x = x[0]
            elif isinstance(x, EmptyResult):
                if isinstance(on_empty, Callable):
                    return on_empty()
                else:
                    return on_empty
            elif isinstance(x, NonEmptyResult):
                x = x.value
            else:
                return x

    def __setitem__(self, key, value):
        self._parent._hash = None
        idx = self._parent._to_explicit_idx(key)
        if isinstance(key, int):
            self._parent._data[self._group][key] = value
        elif hasattr(value, "__len__") and len(idx) == len(value) and not isinstance(value, str):
            for i, v in zip(idx, value):
                self._parent._data[self._group][i] = v
        elif isinstance(value, str) or len(idx) == 1 or not hasattr(value, "__len__"):
            for i in idx:
                self._parent._data[self._group][i] = value
        else:
            raise ValueError("Value to be assigned has different length that given indices!")

    def cost(self, aggregate=True):
        costs = dict()
        for x in self.llm_results:
            for y in x or []:
                costs[y.fingerprint] = y.costs
        return sum(costs.values(), Costs()) if aggregate else costs

    def nonempty_values(self):
        """Return all non-empty values."""
        return [
            x
            for x in self.values
            if not isinstance(x, EmptyResult)
            and not (isinstance(x, list) and len(x) == 1 and isinstance(x[0], EmptyResult))
        ]

    @property
    def empty_rate(self):
        data = self.values
        return len([x for x in data if isinstance(x, EmptyResult)]) / len(data)

    @property
    def timeout_rate(self):
        """Return the fraction of results that are of type `TimeoutResult`."""
        data = self.values
        return len([x for x in data if isinstance(x, TimeoutResult)]) / len(data)

    @property
    def total_cost(self):
        return self.cost().total

    @property
    def input_cost(self):
        return self.cost().input

    @property
    def output_cost(self):
        return self.cost().output

    @property
    def llm_results(self):
        """Return all intermediate LLMResults that were used in the computation of each row."""
        data = self.raw_values
        res = list()
        for x in data:
            res.append(Result.bfs(x, lambda x: isinstance(x, LLMResult)))
        return res

    @property
    def llm_requests(self):
        return [[y.request_text for y in x] if x is not None else None for x in self.llm_results]

    @property
    def llm_responses(self):
        return [[y.value for y in x] if x is not None else None for x in self.llm_results]

    def normalized_values(self, on_empty=dict, ignore=("id",), ensure_list=False, flatten_1d_dicts=True):
        primitives = list()
        for x in self.raw_values:
            unwrapped = self.unwrap(x, on_empty=on_empty, ignore=ignore, flatten_1d_dicts=flatten_1d_dicts)
            if ensure_list and not isinstance(unwrapped, list):
                primitives.append([unwrapped])
            else:
                primitives.append(unwrapped)
        return primitives


class MinibatchIterator:
    def __init__(self, df, minibatch_size):
        self._df = df
        self._minibatch_size = minibatch_size

    def __iter__(self) -> Iterator[list[int]]:
        n_rows = len(self._df)
        if self._minibatch_size == 1:
            return iter(range(n_rows))
        else:
            return more_itertools.chunked_even(range(n_rows), self._minibatch_size)

    def __len__(self) -> int:
        n_rows = len(self._df)
        return math.ceil(n_rows / self._minibatch_size)
