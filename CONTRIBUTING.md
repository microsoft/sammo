# How to contribute

Welcome, and thank you for contributing to the project!

## Mcrosoft Contributor License Agreement

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit <https://cla.opensource.microsoft.com>.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

## Code of Conduct

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com>) with any additional questions or comments.

## Submitting an Issue

Please [search for your issue](https://github.com/microsoft/sammo/issues?q=is%3Aissue) before submitting a new one.

If nothing relevant shows up, please do [open a new issue](https://github.com/microsoft/sammo/issues/new) and provide as much detail as you can (ie: OS, python version, data formats, etc). Outputs of commands, error logs, source code snippets, etc are welcomed and will help to trace down the issue. Questions are also welcomed as they provide an opportunity for us to improve the documentation.

## Setting up your dev environment

This project uses [Poetry](https://python-poetry.org/) for project management. Some tasks have been standardized to execute via the [Poe](https://poethepoet.natn.io/) task runner.

We recommend that you install Poetry using [pipx](https://pipx.pypa.io/stable/) so that it's isolated from the sammo codebase.

```
pipx install poetry
```

Next, check out sammo:

```
# assume HTTPS, adjust for SSH
git clone https://github.com/microsoft/sammo.git
cd sammo
```

Optional, but recommended: have poetry create a venv in the project folder rather than in its cache dir

```
poetry config virtualenvs.in-project true --local
```

Install the dev dependencies

```
poetry install --with dev
```

Show the configured tasks available through the Poe runner:

```
poetry run poe
```

Set up pre-commit hooks

```
poetry run pre-commit install
```

## Running Tests

The [pytest](https://docs.pytest.org/) tests can be run using the following command

```
poetry run poe test
```

arguments can be appended (ie for verbose mode)

```
poetry run poe test -v
```

## Running Type Checks

```
poetry run poe type-check
```

## Building and previewing documentation

This project uses [Jupyter Book](https://jupyterbook.org/) for documentation. The documentation configuration and contents are contained in the `docs` folder.

To build the documentation, run the following command:

```
poetry run poe build-docs
```

to preview it using Python's built-in HTTP server, run:

```
poetry run poe serve-docs
```

This will open a server accessible at http://localhost:8000 to preview the documentation site. You can change the host and port as needed (these arguments just pass through to the call to `http.server`):

```
poetry run poe serve-docs -b 0.0.0.0 8001
```

## PR workflow

All changes must come through a pull request on a feature branch.

1. If there isn't an existing issue for your change, please make one
1. Ensure your local main branch is up to date
1. Create a new branch to hold your changes. Suggested branch naming convention is `<your github user>/<a short description of your changes>`. For example `pbourke/update-contributor-docs`.
1. Run `poetry version`. If the current version is **not** a pre-release (ie 0.1.0.6 vs 0.1.0.6rc0), then bump to the next pre-release version:

   ```
   # example version bump
   $ poetry version
   sammo 0.1.0.6
   $ poetry version 0.1.0.7rc0
   Bumping version from 0.1.0.6 to 0.1.0.7rc0
   ```
1. Make your changes and commit to your feature branch.
1. Push to GitHub as appropriate (to your fork for non-maintainers)
1. Open a Pull Request to the project and reference the associated issue from your PR
1. GitHub Actions will run automated checks and tests
1. When you're ready, request review from the maintainers

## Release Process

The following instructions are for maintainers

1. Each release should begin with a PR to (at the least) update the version from pre-release to final
1. Decide on the new version number by following [Semantic Versioning](https://semver.org/) principles
1. After the release PR is merged, the release can be made from the main branch. Each release is given a tag on the main branch with the version number (this happens automatically via the GH release mechanism)
1. Go to [the sammo project releases page](https://github.com/microsoft/sammo/releases) and click "Draft a new release"
1. Enter the new version number as the tag and release title and give a brief description
1. Click "Publish release"
1. A GitHub Actions release hook will run the automated checks and tests, publish the package to PyPI and publish the documentation to the GitHub Pages site
