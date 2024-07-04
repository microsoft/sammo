import copy
import json
import pathlib

import click
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
import chromadb
import pandas as pd
import dspy
from dspy.retrieve.chromadb_rm import ChromadbRM
from dspy.evaluate import Evaluate
from dspy.teleprompt import MIPRO

BASE_DIR = pathlib.Path(__file__).parent / "rag_tuning_sammo"
RESULTS_DIR = BASE_DIR / "dspy"
RESULTS_DIR.mkdir(exist_ok=True)
DB_PATH = str(BASE_DIR / "chroma")
TASKS = ["smcalflow", "geo880", "overnight"]

DATA_PATHS = {
    "smcalflow": r"C:\Data\smcalflow_cs_simple_v3\all.jsonl",
    "geo880": r"C:\Data\geo880_v2\all.jsonl",
    "overnight": r"C:\Data\overnight_socialnetwork\all.jsonl",
}
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
    "llama-2-alt": {
        "config": {"model": "meta-llama/Llama-2-70b-chat-hf", "api_base": "https://api.together.xyz/v1/"},
        "credentials": CONFIG_PATH / "personal.together",
        "class": "Together",
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
TASKS = ["smcalflow", "geo880", "overnight"]


class GenerateAnswer(dspy.Signature):
    context = dspy.InputField(desc="may contain relevant facts")
    input = dspy.InputField()
    answer = dspy.OutputField()


class RAG(dspy.Module):
    def __init__(
        self,
        n_fewshot=10,
        instructions="Answer questions with short factoid answers.",
    ):
        super().__init__()
        my_module = copy.copy(GenerateAnswer)
        my_module.__doc__ = instructions

        self.retrieve = dspy.Retrieve(k=n_fewshot)
        self.generate_answer = dspy.Predict(my_module)

    def forward(self, input):
        context = self.retrieve(input).passages
        prediction = self.generate_answer(context=context, input=input)
        return dspy.Prediction(context=context, answer=prediction.answer)


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


class TogetherPatched(dspy.OpenAI):
    pass


def normalize(x):
    return x.lower().strip()


def accuracy(gold, pred, trace=None) -> bool:
    return normalize(pred.answer) == normalize(gold.answer)


def load_task(task_id, metadata_path=BASE_DIR.parent):
    with open(metadata_path / "data_splits.json") as f:
        meta_data = json.load(f)[task_id]
    if not pathlib.Path(DATA_PATHS[task_id]).exists():
        raise FileNotFoundError(
            f"Data file {DATA_PATHS[task_id]} not found. "
            "You can download it from https://github.com/allenai/code-semparse/tree/main/datasets"
        )
    full_data = pd.read_json(DATA_PATHS[task_id], lines=True).set_index("qid")
    output = dict(task_id=task_id, constants=meta_data["dsl"])
    for split in ["train", "test", "incontext"]:
        joined = pd.DataFrame({"qid": meta_data[split]}).set_index("qid").join(full_data)
        output[split] = dict(
            data=joined.to_dict(orient="records"),
            constants=meta_data["dsl"],
        )
    return output


def init_retriever(coll_name, docs, overwrite=False):
    client = chromadb.PersistentClient(path=DB_PATH)
    if coll_name in [c.name for c in client.list_collections()]:
        if overwrite:
            client.delete_collection(coll_name)
        else:
            return

    collection = client.create_collection(name=coll_name, embedding_function=EMBEDDING_FUNC)
    collection.add(
        documents=[f"Input: {doc['source']}\nAnswer:{doc['target']}" for doc in docs],
        ids=[str(i) for i in range(len(docs))],
    )


@click.command()
@click.option("--llm", default=MODELS[0], type=click.Choice(MODELS), prompt=True)
@click.option("--task-id", default=TASKS[0], type=click.Choice(TASKS), prompt=True)
@click.option("--uuid", default=None, type=str)
@click.option("--confirmed", is_flag=True, default=None)
@click.option("--debug", default=True, type=bool, prompt=True)
def main(
    llm,
    task_id,
    uuid,
    confirmed,
    num_threads=16,
    show_example=True,
    n_fewshot=10,
    debug=False,
):
    if confirmed is None:
        click.confirm(
            f"Do you want to run {task_id} with {llm}?",
            abort=True,
            default=True,
        )
    task = load_task(task_id)
    model_config = MODEL_CONFIGS[llm]
    config = json.loads(model_config["credentials"].read_text())
    llm_class = {"OpenAI": dspy.OpenAI, "DeepInfra": DeepInfra, "Together": TogetherPatched}[model_config["class"]]
    runner = llm_class(api_key=config["api_key"], **model_config["config"])
    num_threads = 1 if debug else num_threads
    run_id = f"{llm}_{task['task_id']}"

    init_retriever(task["task_id"], task["incontext"]["data"])
    retriever_model = ChromadbRM(
        task["task_id"],
        DB_PATH,
        embedding_function=EMBEDDING_FUNC,
        k=n_fewshot,
    )
    dspy.settings.configure(lm=runner, rm=retriever_model)

    # Tell DSPy that the 'input' field is the input. Any other fields are labels and/or metadata.
    trainset = [dspy.Example(input=x["source"], answer=x["target"]).with_inputs("input") for x in task["train"]["data"]]
    testset = [dspy.Example(input=x["source"], answer=x["target"]).with_inputs("input") for x in task["test"]["data"]]
    if debug:
        trainset = trainset[:5]
        testset = testset[:5]
    dspy_program = RAG(n_fewshot=n_fewshot, instructions=task["constants"]["full_dd"])
    if show_example or debug:
        dspy_program(input=trainset[0].input)
        runner.inspect_history(n=1)

    # Set up a basic teleprompter, which will compile our RAG program.
    teleprompter = MIPRO(
        metric=accuracy,
        num_candidates=2 if debug else 10,
        track_stats=True,
    )

    # Compile!
    optimized_program = teleprompter.compile(
        dspy_program,
        trainset=trainset,
        num_trials=1 if debug else 24,
        max_bootstrapped_demos=1 if debug else 5,
        view_examples=False,
        max_labeled_demos=1 if debug else 5,
        eval_kwargs=dict(num_threads=num_threads, display_progress=True, display_table=0),
        requires_permission_to_run=False,
    )
    runner.inspect_history(n=1)

    for name, parameter in optimized_program.named_predictors():
        print(name)
        print(parameter)

    eval_params = dict(
        metric=accuracy,
        num_threads=num_threads,
        display_progress=True,
        display_table=0,
        return_outputs=True,
    )

    y_test_score, y_test = Evaluate(devset=testset, **eval_params)(optimized_program)
    print(y_test_score)
    runner.inspect_history(n=1)
    y_train_score, y_train = Evaluate(devset=trainset, **eval_params)(optimized_program)
    try:
        logs = str(optimized_program.trial_logs)
        llm = json.dumps(optimized_program.dump_state())
    except Exception as e:
        print("Failed to dump model state.", e)
        logs = str(optimized_program.trial_logs)
        llm = str(optimized_program.dump_state())

    state = json.dumps(
        {
            "y_test_score": y_test_score / 100.0,
            "y_train_score": y_train_score / 100.0,
            "y_test_input": [v[0].toDict() for v in y_test],
            "y_test_output": [v[1].toDict() for v in y_test],
            "y_train_input": [v[0].toDict() for v in y_train],
            "y_train_output": [v[1].toDict() for v in y_train],
            "run_id": run_id,
            "logs": logs,
            "model": llm,
        },
        indent=4,
    )
    (RESULTS_DIR / f"{run_id}.json").write_text(state)


if __name__ == "__main__":
    EMBEDDING_FUNC = OpenAIEmbeddingFunction(
        api_key=json.loads(MODEL_CONFIGS["gpt-3.5"]["credentials"].read_bytes())["api_key"],
        model_name="text-embedding-3-small",
    )
    main()
