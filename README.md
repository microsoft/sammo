# SAMMO ([📘User Guide](https://microsoft.github.io/sammo/))

[![Latest PyPI version](https://img.shields.io/pypi/v/sammo.svg)](https://pypi.python.org/pypi/sammo)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/microsoft/sammo/master?urlpath=tree/tutorials/0_quickstart.ipynb)

A flexible, easy-to-use library for running and optimizing prompts for Large Language Models (LLMs).

![Overview](https://microsoft.github.io/sammo/_images/overview.png)

## How to Get Started
Go to the [user guide](https://microsoft.github.io/sammo/) for examples, how-tos, and API reference.

Just want to have a quick look? Try the [live demo on Binder](https://mybinder.org/v2/gh/microsoft/sammo/master?urlpath=tree/tutorials/0_quickstart.ipynb).

<!--start-->
### Install library only

```bash
pip install sammo
```

### Install and run tutorials

***Prerequisites***
* Python 3.11+

The following commands will install sammo and jupyter and launch jupyter notebook. It's recommended that you create and activate a virtualenv prior to installing packages.

```bash
pip install sammo jupyter

# clone sammo to a local directory
git clone https://github.com/microsoft/sammo.git
cd sammo

# launch jupyter notebook and open tutorials directory
jupyter notebook --notebook-dir docs/tutorials
```

## Use Cases
SAMMO is designed to support
- **Efficient data labeling**: Supports minibatching by packing and parsing multiple datapoints into a single prompt.
- **Prompt prototyping and engineering**: Re-usable components and prompt structures to quickly build and test new prompts.
- **Instruction optimization**: Optimize instructions to do better on a given task.
- **Prompt compression**: Compress prompts while maintaining performance.
- **Large-scale prompt execution**: parallelization
and rate-limiting out-of-the-box so you can run many queries in parallel and at scale without overwhelming the LLM API.

It is less useful if you want to build
- Interactive, agent-based LLM applications (→ check out [AutoGen](https://microsoft.github.io/autogen/))
- Interactive, production-ready LLM applications (→ check out [LangChain](https://www.langchain.com/))


## Example
This is extending the [chat dialog example from Guidance](https://github.com/guidance-ai/guidance#user-content-chat-dialog-notebook) by running queries in parallel.

```python
runner = OpenAIChat(model_id="gpt-3.5-turbo", api_config=API_CONFIG)
expert_names = GenerateText(
    Template(
        "I want a response to the following question:"
        "{{input}}\n"
        "Name 3 world-class experts (past or present) who would be great at answering this? Don't answer the question yet."
    ),
    system_prompt="You are a helpful and terse assistant.",
    randomness=0,
    max_tokens=300,
)

joint_answer = GenerateText(
    "Great, now please answer the question as if these experts had collaborated in writing a joint anonymous answer.",
    history=expert_names,
    randomness=0,
    max_tokens=500,
)

questions = [
    "How can I be more productive?",
    "What will AI look like in 10 years?",
    "How do we end world hunger?",
]
print(Output(joint_answer).run(runner, questions))
```

<!--end-->

## Licence

This project is licensed under [MIT](https://choosealicense.com/licenses/mit/).

## Authors

`SAMMO` was written by [Tobias Schnabel](mailto:sammo@microsoft.com).

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit <https://cla.opensource.microsoft.com>.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com>) with any additional questions or comments.
