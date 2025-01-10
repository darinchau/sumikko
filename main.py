import os
from emulator import Emulator, ImageReference, save_screenshot
import numpy as np
from numpy.typing import NDArray
import logging
from actions import *

def setup_logger():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    logger.addHandler(console_handler)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)

    for name in logging.Logger.manager.loggerDict:
        if not name.startswith("__"):
            logging.getLogger(name).setLevel(logging.WARNING)

    # Set child loggers to debug
    logging.getLogger("emulator").setLevel(logging.DEBUG)
    logging.getLogger("actions").setLevel(logging.DEBUG)
    logging.getLogger("__main__").setLevel(logging.INFO)
    return logger

logger = setup_logger()

def get_master_list_of_action() -> list[Action]:
    return [
        OpenMinigameAction(),
        StartMinigameAction(),
        PlayMinigameAction(),
        MinigameQuitAction(),
        CrossButtonAction(),
        FailedToReadDownloadDataAction(),
        TitleScreenAction(),
        # WordButtonAction("Close"),
        WaitAction(3)
    ]

def run(em: Emulator):
    next_actions = get_master_list_of_action()
    while True:
        for action in next_actions:
            logger.debug(f"Checking if {action} can be performed")
            if action.condition(em):
                logger.info(f"Performing {action}")
                next_actions = action.perform(em)
                break
        else:
            next_actions = get_master_list_of_action()

def main():
    em = Emulator("localhost:21503")
    try:
        run(em)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        logger.exception(e)

if __name__ == "__main__":
    main()
