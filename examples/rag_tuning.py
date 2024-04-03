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

MODEL_IDS = {
    "gpt-3.5": "gpt-3.5-turbo-16k-0613",
    "gpt-4": "gpt-4-0613",
    "llama-2": "meta-llama/Llama-2-70b-chat-hf",
    "mixtral": "cognitivecomputations/dolphin-2.6-mixtral-8x7b",
}
DATA_PATHS = {
    "smcalflow": r"C:\Data\smcalflow_cs_simple_v3\all.jsonl",
    "geo880": r"C:\Data\geo880_v2\all.jsonl",
    "overnight": r"C:\Data\overnight_socialnetwork\all.jsonl",
}

MAIN_FOLDER = sammo.utils.DEFAULT_SAVE_PATH
CONFIG_PATH = MAIN_FOLDER.parent.parent / "config"


def accuracy(y_true: DataTable, y_pred: DataTable) -> EvaluationScore:
    y_true, y_pred = y_true.outputs.normalized_values(on_empty=""), y_pred.outputs.normalized_values(on_empty="")
    mistakes = list()
    for i in range(len(y_true)):
        if y_true[i].lower() != str(y_pred[i]).lower():
            mistakes.append(i)

    return EvaluationScore(1 - len(mistakes) / len(y_true), mistakes)


class SmallSearchSpace:
    def __init__(self, dtrain, examples, embedding_runner):
        self.examples = examples
        self.dtrain = dtrain
        self._embedding_runner = embedding_runner

    def __call__(self, return_raw=False):
        orientation = search_op.one_of(["item", "kind"], name="orientation")
        example_formatter = search_op.one_of(
            [
                QuestionAnswerFormatter(
                    all_labels=self.dtrain.outputs.unique(), orient=orientation, attributes_processor=None
                ),
                XMLDataFormatter(orient=orientation, attributes_processor=None),
                JSONDataFormatter(orient=orientation, attributes_processor=None),
            ]
        )

        instr = search_op.one_of(["full_dd", "list_of_operators"], name="instructions")
        structure = [
            Section("Syntax", f"{self.dtrain.constants[instr]}"),
            Section(
                "Examples",
                EmbeddingFewshotExamples(
                    self._embedding_runner,
                    self.examples,
                    search_op.one_of([10, 5], name="n_examples"),
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


def load_task(task_id):
    with open(MAIN_FOLDER / "data_splits.json") as f:
        meta_data = json.load(f)[task_id]
    if not pathlib.Path(DATA_PATHS[task_id]).exists():
        raise FileNotFoundError(
            f"Data file {DATA_PATHS[task_id]} not found. "
            "You can download it from https://github.com/allenai/code-semparse/tree/main/datasets"
        )
    full_data = pd.read_json(DATA_PATHS[task_id], lines=True).set_index("qid")
    output = dict()
    for split in ["train", "test", "incontext"]:
        joined = pd.DataFrame({"qid": meta_data[split]}).set_index("qid").join(full_data)
        output[split] = DataTable.from_pandas(
            joined, output_fields="target", input_fields="source", constants=meta_data["dsl"]
        )
    return output


def get_runner(model, task_id, low_timeout_tolerance=False, timeout=90):
    model_id = MODEL_IDS[model]
    if model in ["llama-2", "mixtral"]:
        api_config = CONFIG_PATH / "personal.deepinfra"
        equivalence_class = model_id.split("/")[-1]
    elif model.startswith("gpt"):
        api_config = CONFIG_PATH / "personal.openai"
        equivalence_class = "gpt-3.5-turbo-16k" if "gpt-3" in model else model_id
    else:
        raise ValueError("Unknown endpoint")
    run_id = f"{task_id}_{equivalence_class}"
    runner = OpenAIChat(
        model_id=model_id,
        api_config=api_config,
        equivalence_class=equivalence_class,
        rate_limit=10 if "gpt-3" in model else [AtMost(10, "running"), AtMost(2, "rejected", 1)],
        cache=sammo.store.PersistentDict(MAIN_FOLDER / f"{run_id}.cache.tsv"),
        timeout=timeout if "gpt" in model else timeout * 2,
        max_timeout_retries=1 if low_timeout_tolerance else 0,
        max_retries=50000,
    )
    return run_id, runner


@click.group(invoke_without_command=True)
@click.pass_context
def cli(ctx):
    if ctx.invoked_subcommand is None:
        for model in ["gpt-4", "gpt-3.5", "mixtral", "llama-2"]:
            for task_id in ["geo880", "smcalflow", "overnight"]:
                ctx.invoke(main, task_id=task_id, model=model)
    else:
        click.echo(f"I am about to invoke {ctx.invoked_subcommand}")


@cli.command()
@click.option("--model", default="gpt-3.5", type=click.Choice(["gpt-3.5", "mixtral", "gpt-4", "llama-2"]))
@click.option("--task-id", default="smcalflow", type=click.Choice(["smcalflow", "geo880", "overnight"]))
@click.option("--uuid", default=None, type=str)
def main(model, task_id, uuid):
    loaded_data = load_task(task_id)
    d_incontext, d_test = loaded_data["incontext"], loaded_data["test"]
    d_train = loaded_data["train"]

    print("Duplicates in train:", len(d_incontext) - len(d_incontext.inputs.unique()))
    print(f"Dataset sizes: {len(d_train)} (train), {len(d_incontext)} (incontext), {len(d_test)} (test)")

    run_id, runner = get_runner(model, task_id, timeout=30)
    embedder = OpenAIEmbedding(
        model_id="text-embedding-3-small",
        api_config=CONFIG_PATH / "personal.openai",
        rate_limit=10,
        cache=sammo.store.SqlLiteDict(MAIN_FOLDER / f"{task_id}" / "fewshotcache"),
    )
    search_space = SmallSearchSpace(d_train, d_incontext, embedding_runner=embedder)

    # Baseline
    baseline_model = EnumerativeSearch(runner, search_space, accuracy, maximize=True, max_candidates=1)
    baseline_model.fit_transform(d_train)
    dtest_baseline = baseline_model.transform(d_test)
    baseline_model.save(MAIN_FOLDER / "baseline" / f"{run_id}.model.json")

    # SAMMO
    sammo_model = EnumerativeSearch(runner, search_space, accuracy, maximize=True)
    sammo_model.fit(d_train)
    sammo_model.show_report()
    dtest_sammo = sammo_model.transform(d_test)
    sammo_model.save(MAIN_FOLDER / "sammo" / f"{run_id}.model.json")

    print(f"Baseline (test):\n {accuracy(d_test, dtest_baseline)}")
    print(f"SAMMO (test):\n {accuracy(d_test, dtest_sammo)}")


if __name__ == "__main__":
    cli()
