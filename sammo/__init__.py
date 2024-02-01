# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import logging
import beartype
import sammo.utils as utils
from pathlib import Path

PROMPT_LOGGER_NAME = "prompt_logger"


@beartype.beartype
def setup_logger(
    default_level: int | str = "DEBUG",
    log_prompts_to_file: bool = False,
    prompt_level: int | str = "DEBUG",
    prompt_logfile_name: str = None,
) -> logging.Logger:
    if log_prompts_to_file:
        if prompt_logfile_name is None:
            prompt_logfile_name = (utils.MAIN_PATH / "logs" / utils.MAIN_NAME).with_suffix(".log")
        log_prompts_to_file = Path(prompt_logfile_name)
        log_prompts_to_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_prompts_to_file, mode="w", delay=0, encoding="utf-8")
        file_handler.setFormatter(logging.Formatter("===%(asctime)s===\n%(message)s"))

        # add logger just for prompt requests
        prompt_logger = logging.getLogger(PROMPT_LOGGER_NAME)
        prompt_logger.setLevel(prompt_level)
        prompt_logger.addHandler(file_handler)

    logger = logging.getLogger(__name__)
    logger.setLevel(default_level)
    logger.handlers = list()
    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter("%(asctime)s,%(msecs)d: %(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(console)

    return logger
