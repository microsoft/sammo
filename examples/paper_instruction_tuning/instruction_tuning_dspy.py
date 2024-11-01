import copy
import json
import pathlib
import dspy
import orjson
import click

from dspy.evaluate import Evaluate
from dspy.teleprompt import COPRO

CONFIG_PATH = pathlib.Path(__file__).parent.parent.parent / "config"
MODEL_CONFIGS = {
    "gpt-3.5": {
        "config": {"model": "gpt-3.5-turbo-16k-0613"},
        "credentials": CONFIG_PATH / "personal.openai",
        "class": "OpenAI",
    },
    "gpt-4": {
        "config": {"model": "gpt-4-0613"},
        "credentials": CONFIG_PATH / "personal.openai",
        "class": "OpenAI",
    },
    "llama-2": {
        "config": {
            "model": "meta-llama/Llama-2-70b-chat-hf",
            "api_base": "https://api.deepinfra.com/v1/openai/",
        },
        "credentials": CONFIG_PATH / "personal.deepinfra",
        "class": "DeepInfra",
    },
    "mixtral": {
        "config": {
            "model": "cognitivecomputations/dolphin-2.6-mixtral-8x7b",
            "api_base": "https://api.deepinfra.com/v1/openai/",
        },
        "credentials": CONFIG_PATH / "personal.deepinfra",
        "class": "DeepInfra",
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
RESULTS_DIR = pathlib.Path(__file__).parent / "dspy"
RESULTS_DIR.mkdir(exist_ok=True)


class BasicQA(dspy.Signature):
    question = dspy.InputField()
    answer = dspy.OutputField()


class DeepInfra(dspy.OpenAI):
    MAX_BATCH_SIZE = 1

    def __call__(
        self,
        prompt: str,
        **kwargs,
    ):
        n = kwargs.get("n", 1)
        if n > self.MAX_BATCH_SIZE:
            completions = []
            for i in range(0, n, self.MAX_BATCH_SIZE):
                args = dict(**kwargs)
                args["n"] = min(n, i + self.MAX_BATCH_SIZE) - i
                args["temperature"] = kwargs.get("temperature", 0.7) - 0.01 * i
                minibatch = super().__call__(prompt=prompt, **args)
                completions += minibatch
        else:
            completions = super().__call__(prompt=prompt, **kwargs)
        return completions


def normalize(x):
    return x.lower().replace(" ", "")


def accuracy(gold, pred, trace=None) -> bool:
    return normalize(pred.answer) == normalize(gold.answer)


class SimpleTaskPipeline(dspy.Module):
    def __init__(self, instructions):
        super().__init__()

        my_module = copy.copy(BasicQA)
        my_module.__doc__ = instructions
        self.signature = my_module
        self.predictor = dspy.Predict(self.signature)

    def forward(self, question):
        return self.predictor(question=question)


def load_data():
    with open(DATA, "rb") as f:
        splits = orjson.loads(f.read())
    as_dict = dict()
    for task in splits:
        as_dict[task["task_id"]] = task
        for split in ["d_incontext", "d_train", "d_test", "d_val"]:
            as_dict[task["task_id"]][split] = [
                dspy.Example(question=x["input"], answer=x["output"]).with_inputs("question") for x in task[split]
            ]
    return as_dict


def load_program(path):
    loaded_program = SimpleTaskPipeline(None)
    loaded_program.load(path)


@click.command()
@click.option("--llm", default=MODELS[0], type=click.Choice(MODELS), prompt=True)
@click.option("--task-reference_id", default=TASKS[0], type=click.Choice(TASKS), prompt=True)
@click.option("--uuid", default=None, type=str)
@click.option("--confirmed", is_flag=True, default=None)
def main(llm, task_id, uuid, confirmed, num_threads=24, show_example=True):
    if confirmed is None:
        click.confirm(f"Do you want to run {task_id} with {llm}?", abort=True, default=True)
    task = load_data()[task_id]
    model_config = MODEL_CONFIGS[llm]
    config = json.loads(model_config["credentials"].read_text())
    llm_class = {"OpenAI": dspy.OpenAI, "DeepInfra": DeepInfra}[model_config["class"]]
    runner = llm_class(api_key=config["api_key"], **model_config["config"])
    dspy.settings.configure(lm=runner)
    run_id = f"{llm}_{task['task_id']}"

    dspy_program = SimpleTaskPipeline(task["instructions"])

    if show_example:
        pred = dspy_program(question=task["d_train"][0].question)
        runner.inspect_history(n=1)

    copro_teleprompter = COPRO(
        metric=accuracy,
        breadth=12,
        depth=4,
        track_stats=True,
        init_temperature=1.4 if "gpt" in llm else 0.7,
    )

    optimized_program = copro_teleprompter.compile(
        dspy_program,
        trainset=task["d_train"],
        eval_kwargs=dict(num_threads=num_threads, display_progress=True, display_table=0),
    )
    print(optimized_program)

    eval_params = dict(
        metric=accuracy,
        num_threads=num_threads,
        display_progress=True,
        display_table=0,
        return_outputs=True,
    )
    y_test_score, y_test = Evaluate(devset=task["d_test"], **eval_params)(optimized_program)
    print(y_test_score)
    y_train_score, y_train = Evaluate(devset=task["d_train"], **eval_params)(optimized_program)

    state = orjson.dumps(
        {
            "y_test_score": y_test_score / 100.0,
            "y_train_score": y_train_score / 100.0,
            "y_test_input": [v[0].toDict() for v in y_test],
            "y_test_output": [v[1].toDict() for v in y_test],
            "y_train_input": [v[0].toDict() for v in y_train],
            "y_train_output": [v[1].toDict() for v in y_train],
            "run_id": run_id,
            "model": optimized_program.dump_state(),
        },
        option=orjson.OPT_INDENT_2,
    )
    (RESULTS_DIR / f"{run_id}.dspy").write_bytes(state)


if __name__ == "__main__":
    main()
