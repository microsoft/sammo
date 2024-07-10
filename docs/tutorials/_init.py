# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import pathlib
import sammo
from sammo.runners import OpenAIChat
from sammo.base import Template, EvaluationScore
from sammo.components import Output, GenerateText, ForEach, Union
from sammo.extractors import ExtractRegex
from sammo.data import DataTable
import json
import requests
import os

if not "OPENAI_API_KEY" in os.environ:
    raise ValueError("Please set the environment variable 'OPENAI_API_KEY'.")

_ = sammo.setup_logger("WARNING")  # we're only interested in warnings for now

runner = OpenAIChat(
    model_id="gpt-3.5-turbo",
    api_config={"api_key": os.environ["OPENAI_API_KEY"]},
    cache=os.getenv("CACHE_FILE", "cache.tsv"),
    timeout=30,
)


def load_data(
    url="https://github.com/google/BIG-bench/raw/main/bigbench/benchmark_tasks/implicatures/task.json",
):
    task = json.loads(requests.get(url).content)
    # convert label to single string
    for x in task["examples"]:
        x["output"] = max(x["target_scores"], key=x["target_scores"].get)

    return DataTable.from_records(
        task["examples"],
        input_fields="input",
        constants={"instructions": task["task_prefix"]},
    )


def accuracy(y_true: DataTable, y_pred: DataTable) -> EvaluationScore:
    y_true = y_true.outputs.normalized_values()
    y_pred = y_pred.outputs.normalized_values()
    n_correct = sum([y_p == y_t for y_p, y_t in zip(y_pred, y_true)])

    return EvaluationScore(n_correct / len(y_true))
