import click
import sammo
import orjson

from sammo.base import EvaluationScore
from sammo.mutators import (
    BagOfMutators,
    APE,
    InduceInstructions,
    SyntaxTreeMutator,
    APO,
    Paraphrase,
)
from sammo.runners import OpenAIChat
from sammo.throttler import AtMost

logger = sammo.setup_logger(log_prompts_to_file=True)

from sammo import search_op
from sammo.data import DataTable
from sammo.instructions import MetaPrompt, Paragraph, InputData
from sammo.components import Output
from sammo.dataformatters import PlainFormatter
from sammo.search import EnumerativeSearch, BeamSearch
from sammo.store import PersistentDict

import pathlib

MAIN_FOLDER = sammo.utils.DEFAULT_SAVE_PATH
CONFIG_PATH = MAIN_FOLDER.parent.parent.parent / "config"
MODEL_CONFIGS = {
    "gpt-3.5": {
        "full_id": "gpt-3.5-turbo-16k-0613",
        "equivalence_class": "gpt-3.5-turbo-16k",
        "credentials": CONFIG_PATH / "personal.openai",
        "rate_limit": 10,
        "timeout": 90,
        "max_context_window": None,
    },
    "gpt-4": {
        "full_id": "gpt-4-0613",
        "equivalence_class": "gpt-4-0613",
        "credentials": CONFIG_PATH / "personal.openai",
        "rate_limit": 10,
        "timeout": 90,
        "max_context_window": None,
    },
    "llama-2": {
        "full_id": "meta-llama/Llama-2-70b-chat-hf",
        "equivalence_class": "meta-llama/Llama-2-70b-chat-hf",
        "credentials": CONFIG_PATH / "personal.deepinfra",
        "rate_limit": [AtMost(10, "running"), AtMost(2, "rejected", 1)],
        "timeout": 180,
        "max_context_window": 4096,
    },
    "mixtral": {
        "full_id": "cognitivecomputations/dolphin-2.6-mixtral-8x7b",
        "equivalence_class": "dolphin-2.6-mixtral-8x7b",
        "credentials": CONFIG_PATH / "personal.deepinfra",
        "rate_limit": [AtMost(10, "running"), AtMost(2, "rejected", 1)],
        "timeout": 180,
        "max_context_window": None,
    },
}
MODELS = list(MODEL_CONFIGS.keys())
TASKS = [
    "implicatures",
    "metaphor_boolean",
    "navigate",
    "presuppositions_as_nli",
    "sports_understanding",
    "vitaminc_fact_verification",
    "winowhy",
    "word_sorting",
]
DATA = "data_splits.json"
METHODS = ["sammo", "apo", "ape", "grips"]


def accuracy(y_true: DataTable, y_pred: DataTable) -> EvaluationScore:
    def normalize(x):
        if isinstance(x, dict):
            print(x)
        return x.lower().replace(" ", "")

    mistakes = list()

    y_in = y_true.inputs.raw_values
    y_true, y_pred = y_true.outputs.normalized_values(), y_pred.outputs.normalized_values(on_empty="")

    for i in range(len(y_true)):
        is_mistake = normalize(y_true[i]) != normalize(y_pred[i])
        is_mistake = is_mistake and normalize(y_in[i] + y_true[i]) != normalize(y_pred[i])
        if is_mistake:
            mistakes.append(i)

    accuracy = 1 - len(mistakes) / len(y_true)
    return EvaluationScore(accuracy, mistakes)


class InstructionTuningSearchSpace:
    def __init__(self, dtrain):
        self.dtrain = dtrain

    def __call__(self):
        example_formatter = PlainFormatter(all_labels=self.dtrain.outputs.unique(), orient="item")

        labels = self.dtrain.outputs.unique()
        instructions = MetaPrompt(
            [
                Paragraph("Instructions: "),
                Paragraph(
                    search_op.one_of(
                        [
                            self.dtrain.constants["instructions"],
                            "",
                            "Find the best output label given the input.",
                            self.dtrain.constants["instructions"] * 2,
                        ]
                    ),
                    id="instructions",
                ),
                Paragraph("\n"),
                Paragraph(f"Output labels: {', '.join(labels)}\n" if len(labels) <= 10 else ""),
                Paragraph(InputData()),
                Paragraph("Output: "),
            ],
            render_as="raw",
            data_formatter=example_formatter,
        )

        return Output(
            instructions.with_extractor("raise"),
            minibatch_size=1,
            on_error="empty_result",
        )


@click.command()
@click.option("--llm", default=MODELS[0], type=click.Choice(MODELS), prompt=True)
@click.option("--task-id", default=TASKS[0], type=click.Choice(TASKS), prompt=True)
@click.option("--method", default=METHODS[0], type=click.Choice(METHODS), prompt=True)
@click.option("--uuid", default=None, type=str)
@click.option("--confirmed", is_flag=True, default=None)
def main(llm, task_id, method, uuid=None, confirmed=None, debug=False):
    if confirmed is None:
        click.confirm(f"Do you want to run {task_id} with {llm}?", abort=True, default=True)
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
        max_context_window=model_config["max_context_window"],
    )
    all_tasks = {x["task_id"]: x for x in orjson.loads(pathlib.Path(DATA).read_bytes())}
    task = all_tasks[task_id]

    data = dict()
    for k, v in task.items():
        if k.startswith("d_"):
            data[k] = DataTable.from_records(v, constants=dict(instructions=task["instructions"]))

    search_space = InstructionTuningSearchSpace(data["d_train"])
    baseline_performance = EnumerativeSearch(runner, search_space, accuracy, max_candidates=1)
    baseline_performance.fit_transform(data["d_train"])
    baseline_performance.transform(data["d_test"])
    baseline_performance.show_report()
    baseline_performance.save(MAIN_FOLDER / "baseline" / f"{run_id}.model.json")

    if method == "ape":
        prompt_optimizer = BeamSearch(
            runner,
            APE({"id": "instructions"}, search_space, data["d_train"], 5),
            accuracy,
            maximize=True,
            n_initial_candidates=12,
            depth=3,
            mutations_per_beam=2,
            beam_width=4,
            add_previous=True,
        )
    elif method == "apo":
        prompt_optimizer = BeamSearch(
            runner,
            APO(
                {"id": "instructions", "_child": "content"},
                search_space,
                num_gradients=2,
                steps_per_gradient=1,
                num_rewrites=1,
            ),
            accuracy,
            maximize=True,
            depth=7,
            mutations_per_beam=2,
            beam_width=4,
            add_previous=True,
        )
    elif method == "grips":
        mutation_operators = SyntaxTreeMutator(
            {"id": "instructions"},
            search_space,
            PersistentDict(MAIN_FOLDER / "trees" / f"{run_id}.cache.json"),
        )
        prompt_optimizer = BeamSearch(
            runner,
            mutation_operators,
            accuracy,
            maximize=True,
            depth=7,
            mutations_per_beam=2,
            n_initial_candidates=1,
            beam_width=4,
            add_previous=True,
        )
    elif method == "sammo":
        mutation_operators = BagOfMutators(
            search_space,
            InduceInstructions({"id": "instructions"}, data["d_incontext"]),
            APO(
                {"id": "instructions", "_child": "content"},
                None,
                num_gradients=2,
                steps_per_gradient=1,
                num_rewrites=0,
            ),
            Paraphrase({"id": "instructions"}),
            sample_for_init_candidates=True,
        )
        prompt_optimizer = BeamSearch(
            runner,
            mutation_operators,
            accuracy,
            maximize=True,
            depth=4,
            mutations_per_beam=2,
            n_initial_candidates=4,
            beam_width=4,
            add_previous=True,
        )
    prompt_optimizer.fit(data["d_train"])
    prompt_optimizer.show_report()

    if not debug:
        dtest_pred = prompt_optimizer.transform(data["d_test"])
        print(f"Test score: {accuracy(data['d_test'], dtest_pred)}")
    prompt_optimizer.save(MAIN_FOLDER / method / f"{run_id}.model.json")


if __name__ == "__main__":
    main()
