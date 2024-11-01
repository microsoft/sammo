import copy
import json

import click
import spacy
import utils
from sammo.base import EvaluationScore, Runner
from sammo.mutators import *
from sammo.runners import OpenAIChat
from sammo.throttler import AtMost

logger = sammo.setup_logger(log_prompts_to_file=True)

from sammo import search_op
from sammo.data import DataTable
from sammo.instructions import *
from sammo.components import *
from sammo.dataformatters import (
    JSONDataFormatter,
    MultiLabelFormatter,
    QuestionAnswerFormatter,
    XMLDataFormatter,
)
import pathlib

from sammo.search import (
    EnumerativeSearch,
    BeamSearch,
    SequentialSearch,
)

from sammo.store import PersistentDict

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
        "minibatch_size": 5,
    },
    "gpt-4": {
        "full_id": "gpt-4-0613",
        "equivalence_class": "gpt-4-0613",
        "credentials": CONFIG_PATH / "personal.openai",
        "rate_limit": 10,
        "timeout": 90,
        "max_context_window": None,
        "minibatch_size": 10,
    },
    "llama-2": {
        "full_id": "meta-llama/Llama-2-70b-chat-hf",
        "equivalence_class": "meta-llama/Llama-2-70b-chat-hf",
        "credentials": CONFIG_PATH / "personal.deepinfra",
        "rate_limit": [AtMost(10, "running"), AtMost(2, "rejected", 1)],
        "timeout": 180,
        "max_context_window": 4096,
        "minibatch_size": 1,
    },
    "mixtral": {
        "full_id": "cognitivecomputations/dolphin-2.6-mixtral-8x7b",
        "equivalence_class": "dolphin-2.6-mixtral-8x7b",
        "credentials": CONFIG_PATH / "personal.deepinfra",
        "rate_limit": [AtMost(10, "running"), AtMost(2, "rejected", 1)],
        "timeout": 180,
        "max_context_window": None,
        "minibatch_size": 5,
    },
}
MODELS = list(MODEL_CONFIGS.keys())
DATA = "data_splits.json"
METHODS = ["baseline", "rewrite", "stopwords", "syntax", "sammo", "ape"]
TASKS = [
    "task018_mctaco_temporal_reasoning_presence",
    "task019_mctaco_temporal_reasoning_category",
    "task108_contextualabusedetection_classification",
    "task133_winowhy_reason_plausibility_detection",
    "task136_winowhy_knowledge_categorization",
    "task204_mnli_same_genre_classification",
    "task211_logic2text_classification",
    "task212_logic2text_classification",
    "task248_dream_classification",
    "task346_hybridqa_classification",
]


class CompressionObjective:
    def __init__(self, weights=None, tolerance=0.02, min_parse_correctness=0.9):
        self._tolerance = tolerance
        self._weights = weights or dict(input_costs=-1, output_costs=-2)
        self._reference_point = dict()
        self._min_parse_correctness = min_parse_correctness

    def __call__(self, y_true: DataTable, y_pred: DataTable) -> EvaluationScore:
        disagreements = self.exact_disagreements(y_true, y_pred)
        objectives = {
            "accuracy": 1 - len(disagreements) / len(y_true),
            "input_costs": y_pred.outputs.input_cost,
            "output_costs": y_pred.outputs.output_cost,
            "parse_correctness": 1 - y_pred.outputs.empty_rate,
            "weighted_costs": abs(
                y_pred.outputs.input_cost * self._weights["input_costs"]
                + y_pred.outputs.output_cost * self._weights["output_costs"]
            ),
        }
        objective = objectives["weighted_costs"]
        if objectives["accuracy"] + self._tolerance < self._reference_point.get("accuracy", 0):
            objective = float("inf")
        if objectives["parse_correctness"] < self._min_parse_correctness:
            objective = float("inf")
        return EvaluationScore(objective, disagreements, objectives)

    def accuracy(self, y_true: DataTable, y_pred: DataTable) -> EvaluationScore:
        scored = self(y_true, y_pred)
        scored.score = scored.details["accuracy"]
        return scored

    def callibrate(self, y_true: DataTable, y_pred: DataTable):
        scored = self(y_true, y_pred)
        self._reference_point = scored.details

    @staticmethod
    def exact_disagreements(y_true, y_pred):
        y_true, y_pred = y_true.outputs.normalized_values(), y_pred.outputs.normalized_values()
        mistakes = list()
        for i in range(len(y_true)):
            if y_true[i] != y_pred[i]:
                mistakes.append(i)

        return mistakes


class SmallSearchSpace:
    def __init__(self, dtrain, examples, minibatch_size):
        self.examples = examples
        self.dtrain = dtrain
        self.minibatch_size = minibatch_size

    def with_minibatch_size(self, new_size):
        clone = copy.copy(self)
        clone.minibatch_size = new_size
        return clone

    def __call__(self, return_raw=False):
        if self.minibatch_size == "all":
            minibatch_size = search_op.one_of([1, 5, 10], reference_id="minibatch_size")
        else:
            minibatch_size = self.minibatch_size

        example_formatter = search_op.one_of(
            [
                QuestionAnswerFormatter(
                    all_labels=self.dtrain.outputs.unique(),
                    orient="kind",
                    attributes_processor=None,
                ),
                MultiLabelFormatter(all_labels=self.dtrain.outputs.unique(), attributes_processor=None),
                XMLDataFormatter(orient="kind", attributes_processor=None),
                JSONDataFormatter(orient="kind", attributes_processor=None),
            ]
        )
        structure = [
            Section("Task", self.dtrain.constants["instructions"], reference_id="task"),
            Section("Examples", FewshotExamples(self.examples), reference_id="examples"),
            Section(
                "Complete and output in the same format as above",
                InputData(id_offset=len(self.examples)),
            ),
        ]
        instructions = MetaPrompt(
            structure,
            render_as="markdown",
            data_formatter=example_formatter,
        )
        return Output(
            instructions.with_extractor("empty_result"),
            minibatch_size=minibatch_size,
            on_error="empty_result",
        )


class RewrittenSearchSpace(SmallSearchSpace):
    def __init__(self, dtrain, examples, minibatch_size):
        super().__init__(dtrain, examples, minibatch_size)
        _, self.runner = utils.get_api_config("gpt-4", "task_rewriter")

    def __call__(self, return_raw=False):
        messages = [
            "Could you please rephrase the paragraph to make it short, and keep 5% tokens?",
            "Condense the passage to retain only 5% of its original tokens, while preserving its meaning.",
            "Shorten the sentences to 200 tokens.",
            "Trim the text down to 200 tokens in total.",
            "Please provide a concise summary of the given examples in several sentences, ensuring that all reasoning information is included.",
            "Summarize the provided examples in a few sentences, maintaining all essential reasoning aspects.",
            "Remove redundancy and express the text concisely in English, ensuring that all key information and reasoning processes are preserved.",
            "Eliminate repetitive elements and present the text concisely, ensuring that key details and logical processes are retained.",
            "Follow these steps to shorten the given text content: 1. First, calculate the amount of information contained in each sentence, and remove sentences with less information. 2. Next, further condense the text by removing stop words, unnecessary punctuation, and redundant expressions. Refine the content while ensuring that all key information is retained. Let’s do it step by step.",
            "To shorten the given text, follow these steps: a) Determine the information value of each sentence and remove those with lower value. b) Further reduce the text by removing stop words, unneeded punctuation, and superfluous expressions, while making sure to keep all vital information intact. Let’s do it step by step.",
        ]
        minibatch_size = self.minibatch_size
        structure = [
            Section("Task", self.dtrain.constants["instructions"]),
            Section("Examples", FewshotExamples(self.examples)),
        ]
        instructions = MetaPrompt(
            structure,
            render_as="markdown",
            data_formatter=QuestionAnswerFormatter(
                all_labels=self.dtrain.outputs.unique(), orient="kind", attributes_processor=None
            ),
        )
        result = Output(instructions).run(self.runner).outputs.raw_values[0].value
        rewritten = (
            Output(GenerateText(result, system_prompt=search_op.one_of(messages, reference_id="message")))
            .run(self.runner)
            .outputs.raw_values[0]
            .value
        )
        instructions = MetaPrompt(
            [
                Paragraph(rewritten),
                Section(
                    "Complete and output in the same format as above",
                    InputData(id_offset=len(self.examples)),
                ),
            ],
            render_as="markdown",
            data_formatter=QuestionAnswerFormatter(
                all_labels=self.dtrain.outputs.unique(), orient="kind", attributes_processor=None
            ),
        )
        return Output(
            instructions.with_extractor("empty_result"),
            minibatch_size=minibatch_size,
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

    task_info = json.loads(pathlib.Path(DATA).read_bytes())[task_id]
    data = {k: DataTable.from_json(v) for k, v in task_info["data"].items()}
    task_id = task_info["task_id"]

    model_config = MODEL_CONFIGS[llm]
    minibatch_size = model_config["minibatch_size"]
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

    objective = CompressionObjective()
    search_space = SmallSearchSpace(data["d_train"], data["d_incontext"], minibatch_size)

    baseline_model = EnumerativeSearch(runner, search_space, objective, max_candidates=1)
    d_train_pred = baseline_model.fit_transform(data["d_train"])
    dtest_baseline = baseline_model.transform(data["d_test"])
    baseline_model.save_json(
        MAIN_FOLDER / "autobatch" / f"baseline" / f"{run_id}.batchsize={minibatch_size}.model.json"
    )

    objective.callibrate(data["d_train"], d_train_pred)
    ops = {
        "struct": [
            ChangeDataFormat(
                "data_formatter",
                [
                    MultiLabelFormatter(all_labels=data["d_train"].outputs.unique()),
                    JSONDataFormatter(),
                    XMLDataFormatter(),
                ],
            ),
            ChangeDataFormat(
                "data_formatter",
                [
                    MultiLabelFormatter(all_labels=data["d_train"].outputs.unique()),
                    JSONDataFormatter(include_ids=False),
                    XMLDataFormatter(include_ids=False),
                ],
            ),
            ChangeSectionsFormat("sections_format", ["markdown", "xml"]),
        ],
        "examples": [DecreaseInContextExamples(data["d_incontext"], reduction_factor=0.9)],
        "drop": [DropExamples("#examples"), DropIntro("#task")],
        "rewrite": [
            InduceInstructions("#task", data["d_incontext"]),
            ShortenSegment("#task", reduction_factor=0.75),
            ShortenSegment("#task", reduction_factor=0.5),
            SegmentToBulletPoints("#task"),
            RemoveStopWordsFromSegment("#task", [StopwordsCompressor("reuters"), StopwordsCompressor("spacy")]),
            ReplaceParameter(
                r"attributes_processor",
                [None, StopwordsCompressor("reuters"), StopwordsCompressor("spacy")],
            ),
        ],
    }

    mutation_operators = BagOfMutators(
        search_space,
        *sum(ops.values(), []),
        sample_for_init_candidates=False,
    )
    if method == "baseline":
        return
    elif method == "rewrite":
        prompt_optimizer = EnumerativeSearch(
            runner,
            RewrittenSearchSpace(data["d_train"], data["d_incontext"], minibatch_size),
            objective,
            maximize=False,
        )
    elif method == "stopwords":

        def search_space_mutators():
            return [
                RemoveStopWordsFromSegment(
                    "#task",
                    search_op.one_of([StopwordsCompressor("reuters"), StopwordsCompressor("spacy")]),
                ),
                ReplaceParameter(
                    r"attributes_processor",
                    search_op.one_of([None, StopwordsCompressor("reuters"), StopwordsCompressor("spacy")]),
                ),
            ]

        prompt_optimizer = EnumerativeSearch(
            runner,
            search_space_mutators,
            objective,
            maximize=False,
            mutate_from=search_op.get_first_point(search_space),
        )
    elif method == "syntax":
        prompt_optimizer = SequentialSearch(
            runner,
            PruneSyntaxTree(
                "#task",
                search_space,
                objective.accuracy,
                cache=PersistentDict(MAIN_FOLDER / "trees" / f"{run_id}.cache.json"),
            ),
            objective,
            maximize=False,
            depth=48,
        )
    elif method == "sammo":
        prompt_optimizer = BeamSearch(
            runner,
            mutation_operators,
            objective,
            maximize=False,
            depth=6,
            beam_width=4,
            mutations_per_beam=2,
            n_initial_candidates=4,
            add_previous=True,
            max_evals=48,
        )
    elif method == "ape":
        prompt_optimizer = BeamSearch(
            runner,
            APE("#task", search_space, data["d_train"], 5),
            objective,
            maximize=False,
            n_initial_candidates=12,
            depth=3,
            mutations_per_beam=2,
            beam_width=6,  # 12 + 12*2 = 36 from first stage, then 8+8 from second and third, so 36+16 = 52
            add_previous=True,
            max_evals=48,
        )

    d_train_pred = prompt_optimizer.fit_transform(data["d_train"])
    print(f"{method} (train):", objective(data["d_train"], d_train_pred))
    print("Break even: ", prompt_optimizer.break_even(baseline_model.best["costs"]))
    prompt_optimizer.show_report()
    print("Best: ", prompt_optimizer.best)

    if not debug:
        objective.callibrate(data["d_test"], dtest_baseline)
        dtest_pred = prompt_optimizer.transform(data["d_test"])
        print(f"Baseline (test):\n {objective(data['d_test'], dtest_baseline)}")
        print(f"{method} (test):\n {objective(data['d_test'], dtest_pred)}")
    prompt_optimizer.save_json(MAIN_FOLDER / "autobatch" / method / f"{run_id}.model.json")


def check_spacy():
    if not spacy.util.is_package("en_core_web_sm"):
        print("Space model 'en_core_web_sm' not found. Installing...")
        spacy.cli.download("en_core_web_sm")


if __name__ == "__main__":
    check_spacy()
    main()
