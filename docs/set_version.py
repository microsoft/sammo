# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


def setup(app):
    """Reads project version number from pyproject toml and sets it in the Sphinx context
    under the expected key"""
    from poetry.core.factory import Factory

    # read project version string from pyproject.toml
    poetry = Factory().create_poetry()
    version = poetry.package.pretty_version

    # wire up a handler to set the version string in the Sphinx config object after config is initialized
    def set_version_handler(_, config):
        config["version"] = version

    app.connect("config-inited", set_version_handler)

    return {
        "version": "0.1",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
