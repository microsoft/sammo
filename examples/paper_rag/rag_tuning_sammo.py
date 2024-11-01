import pathlib

from sammo.base import EvaluationScore
from sammo.mutators import *
from sammo.runners import OpenAIEmbedding, OpenAIChat
from sammo.throttler import AtMost

logger = sammo.setup_logger(log_prompts_to_file=True)
import pandas as pd
import click
import sammo.store
from sammo import search_op
from sammo.instructions import *
from sammo.components import *
from sammo.dataformatters import JSONDataFormatter, QuestionAnswerFormatter, XMLDataFormatter
import json
from sammo.search import (
    EnumerativeSearch,
)

MAIN_FOLDER = sammo.utils.DEFAULT_SAVE_PATH
CONFIG_PATH = MAIN_FOLDER.parent.parent.parent / "config"
MODEL_CONFIGS = {
    "gpt-3.5": {
        "full_id": "gpt-3.5-turbo-16k-0613",
        "equivalence_class": "gpt-3.5-turbo-16k",
        "credentials": CONFIG_PATH / "personal.openai",
        "rate_limit": 10,
        "timeout": 90,
    },
    "gpt-4": {
        "full_id": "gpt-4-0613",
        "equivalence_class": "gpt-4-0613",
        "credentials": CONFIG_PATH / "personal.openai",
        "rate_limit": 10,
        "timeout": 90,
    },
    "llama-2": {
        "full_id": "meta-llama/Llama-2-70b-chat-hf",
        "equivalence_class": "Llama-2-70b-chat-hf",
        "credentials": CONFIG_PATH / "personal.deepinfra",
        "rate_limit": [AtMost(10, "running"), AtMost(2, "rejected", 1)],
        "timeout": 180,
    },
    "mixtral": {
        "full_id": "cognitivecomputations/dolphin-2.6-mixtral-8x7b",
        "equivalence_class": "dolphin-2.6-mixtral-8x7b",
        "credentials": CONFIG_PATH / "personal.deepinfra",
        "rate_limit": [AtMost(10, "running"), AtMost(2, "rejected", 1)],
        "timeout": 180,
    },
}
MODELS = list(MODEL_CONFIGS.keys())
DATA = MAIN_FOLDER.parent / "data_splits_new.json"
TASKS = ["smcalflow", "geo880", "overnight"]


def accuracy(y_true: DataTable, y_pred: DataTable) -> EvaluationScore:
    y_true, y_pred = y_true.outputs.normalized_values(on_empty=""), y_pred.outputs.normalized_values(on_empty="")
    mistakes = list()
    for i in range(len(y_true)):
        if y_true[i].lower() != str(y_pred[i]).lower():
            mistakes.append(i)

    return EvaluationScore(1 - len(mistakes) / len(y_true), mistakes)


class RagSearchSpace:
    def __init__(self, dtrain, examples, embedding_runner):
        self.examples = examples
        self.dtrain = dtrain
        self._embedding_runner = embedding_runner

    def __call__(self, return_raw=False):
        orientation = search_op.one_of(["item", "kind"], reference_id="orientation")
        example_formatter = search_op.one_of(
            [
                QuestionAnswerFormatter(
                    all_labels=self.dtrain.outputs.unique(), orient=orientation, attributes_processor=None
                ),
                XMLDataFormatter(orient=orientation, attributes_processor=None),
                JSONDataFormatter(orient=orientation, attributes_processor=None),
            ]
        )

        instr = search_op.one_of(["full_dd", "list_of_operators"], reference_id="instructions")
        structure = [
            Section("Syntax", f"{self.dtrain.constants[instr]}"),
            Section(
                "Examples",
                EmbeddingFewshotExamples(
                    self._embedding_runner,
                    self.examples,
                    search_op.one_of([10, 5], reference_id="n_examples"),
                    budget="relative",
                ),
            ),
            Section(
                "Complete and output in the same format as above",
                InputData(id_offset=len(self.examples)),
            ),
        ]
        instructions = MetaPrompt(structure, render_as="markdown", data_formatter=example_formatter)
        return Output(instructions.with_extractor("empty_result"), minibatch_size=1, on_error="empty_result")


def load_task(task_id, data_path=DATA):
    task_info = json.loads(pathlib.Path(data_path).read_bytes())[task_id]
    return {k: DataTable.from_json(v) for k, v in task_info.items()}


@click.command()
@click.option("--llm", default=MODELS[0], type=click.Choice(MODELS), prompt=True)
@click.option("--task-id", default=TASKS[0], type=click.Choice(TASKS), prompt=True)
@click.option("--uuid", default=None, type=str)
@click.option("--confirmed", is_flag=True, default=None)
def main(llm, task_id, uuid, confirmed):
    if confirmed is None:
        click.confirm(f"Do you want to run {task_id} with {llm}?", abort=True, default=True)
    loaded_data = load_task(task_id)
    d_incontext, d_test = loaded_data["incontext"], loaded_data["test"]
    d_train = loaded_data["train"]

    print("Duplicates in train:", len(d_incontext) - len(d_incontext.inputs.unique()))
    print(f"Dataset sizes: {len(d_train)} (train), {len(d_incontext)} (incontext), {len(d_test)} (test)")

    model_config = MODEL_CONFIGS[llm]
    run_id = f"{task_id}_{model_config['equivalence_class'].replace('/', '_')}"
    runner = OpenAIChat(
        model_id=model_config["full_id"],
        api_config=model_config["credentials"],
        equivalence_class=model_config["equivalence_class"],
        rate_limit=model_config["rate_limit"],
        cache=sammo.store.PersistentDict(MAIN_FOLDER / f"{run_id}.cache.tsv"),
        timeout=model_config["timeout"],
        max_retries=50000,
    )

    embedder = OpenAIEmbedding(
        model_id="text-embedding-3-small",
        api_config=CONFIG_PATH / "personal.openai",
        rate_limit=10,
        cache=sammo.store.SqlLiteDict(MAIN_FOLDER / f"{task_id}" / "fewshotcache"),
    )
    search_space = RagSearchSpace(d_train, d_incontext, embedding_runner=embedder)

    # Baseline
    baseline_model = EnumerativeSearch(runner, search_space, accuracy, maximize=True, max_candidates=1)
    baseline_model.fit_transform(d_train)
    dtest_baseline = baseline_model.transform(d_test)
    baseline_model.save_json(MAIN_FOLDER / "baseline" / f"{run_id}.model.json")

    # SAMMO
    sammo_model = EnumerativeSearch(runner, search_space, accuracy, maximize=True)
    sammo_model.fit(d_train)
    sammo_model.show_report()
    dtest_sammo = sammo_model.transform(d_test)
    sammo_model.save_json(MAIN_FOLDER / "sammo" / f"{run_id}.model.json")

    print(f"Baseline (test):\n {accuracy(d_test, dtest_baseline)}")
    print(f"SAMMO (test):\n {accuracy(d_test, dtest_sammo)}")


if __name__ == "__main__":
    main()
