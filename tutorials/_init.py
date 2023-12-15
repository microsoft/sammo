import pathlib
import sammo
from sammo.runners import OpenAIChat
from sammo.base import Template, EvaluationScore
from sammo.components import Output, GenerateText, ForEach, Union
from sammo.extractors import ExtractRegex
from sammo.data import DataTable
import json
import requests

API_CONFIG_FILE = pathlib.Path().cwd().parent / "config" / "personal.openai"
API_CONFIG = ""
if API_CONFIG_FILE.exists():
    API_CONFIG = API_CONFIG_FILE
if not API_CONFIG:
    raise ValueError('Please set API_CONFIG to {"api_key": "YOUR_KEY"}')

_ = sammo.setup_logger("WARNING")  # we're only interested in warnings for now

runner = OpenAIChat(
    model_id="gpt-3.5-turbo-16k",
    api_config=API_CONFIG,
    cache=CACHE_FILE,
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