import os
from emulator import Emulator, ImageReference, save_screenshot
import numpy as np
from numpy.typing import NDArray
import logging
from minigame import *
import time
import random

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
    return logger

logger = setup_logger()

class SummikoEmulator(Emulator):
    @property
    def _minigame_reference(self):
        if not self.screen_size == (1600, 900):
            raise RuntimeError("Screen size is not 1600x900")
        return ImageReference(1060, 40, 810, 50, "references/minigame_reference.png")

    @property
    def _minigame_start_reference(self):
        if not self.screen_size == (1600, 900):
            raise RuntimeError("Screen size is not 1600x900")
        return ImageReference(790, 150, 710, 20, "references/minigame_start_reference.png")

    @property
    def _minigame_screen_reference(self):
        if not self.screen_size == (1600, 900):
            raise RuntimeError("Screen size is not 1600x900")
        return ImageReference(792, 100, 99, 20, "references/minigame_screen_reference.png")

    def _get_current_cross_button(self):
        return self.find_reference("references/cross-button.png")

    def open_minigames_screen(self):
        logger.debug("Trying to open minigame screen")

        if self.has_reference(self._minigame_screen_reference):
            logger.debug("Minigame screen already open")
            return True

        if not self.has_reference(self._minigame_reference):
            logger.debug("Minigame button not found.")
            cross_button = self._get_current_cross_button()
            if cross_button is not None:
                logger.debug(f"Found cross button at {cross_button.x_center}, {cross_button.y_center}")
                self.tap(cross_button)
                return self.open_minigames_screen()
            logger.info("Minigame not available")
            return False

        self.tap(self._minigame_reference.x_center, self._minigame_reference.y_center)
        logger.info("Opened minigame screen")
        return True

    def start_minigame(self):
        logger.debug("Trying to start minigame")
        if self.has_reference(self._minigame_start_reference):
            self.tap(self._minigame_start_reference.x_center, self._minigame_start_reference.y_center)
            logger.info("Started minigame screen")
            return True
        logger.info("Minigame start button not found")
        return False

    def solve_minigame(self, add_random_time_delay: bool = True):
        logger.debug("Solving minigame")
        grid = detect_grid(self.screencap())
        logger.debug("Detected grid")
        logger.debug(show_grid(grid))

        if not is_valid_grid(grid):
            logger.error("Invalid grid")
            return False

        solution = solve_grid(grid)
        if solution is None:
            logger.error("No solution found")
            return False

        logger.debug("Found solution")
        for p0, p1, p2, p3 in solution:
            logger.debug(f"({p0}, {p1}) -> ({p2}, {p3})")

        if solution is not None:
            for p0, p1, p2, p3 in solution:
                if add_random_time_delay:
                    time.sleep(random.random() * 0.5)
                self.tap(get_grid_reference(p0, p1))
                if add_random_time_delay:
                    time.sleep(random.random() * 0.5)
                self.tap(get_grid_reference(p2, p3))
        logger.info("Solved minigame")
        return True

def main():
    em = SummikoEmulator("localhost:21503")
    take_grid_screenshot(em.screencap())

if __name__ == "__main__":
    main()
