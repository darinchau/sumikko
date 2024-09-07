import os
from emulator import Emulator, ImageReference, save_screenshot
import numpy as np
from numpy.typing import NDArray
import logging
from minigame import detect_grid, get_grid_reference, show_grid, solve_grid, InvalidGrid
import time
import random
from model import ImageClassifier

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

    @property
    def _minigame_ingame_reference(self):
        if not self.screen_size == (1600, 900):
            raise RuntimeError("Screen size is not 1600x900")
        return ImageReference(1380, 40, 100, 40, "references/minigame_ingame_reference.png")

    @property
    def _minigame_quit_reference(self):
        if not self.screen_size == (1600, 900):
            raise RuntimeError("Screen size is not 1600x900")
        return ImageReference(1000, 100, 600, 20, "references/minigame_quit_button.png")

    @property
    def _minigame_screen_cross_reference(self):
        if not self.screen_size == (1600, 900):
            raise RuntimeError("Screen size is not 1600x900")
        return ImageReference(1521, 20, 136, 20, "references/minigame_screen_cross_button.png")

    @property
    def _minigame_ingame_cross_reference(self):
        if not self.screen_size == (1600, 900):
            raise RuntimeError("Screen size is not 1600x900")
        return ImageReference(1530, 20, 80, 20, "references/minigame_cross_button.png")

    @property
    def _minigame_retry_cross_reference(self):
        if not self.screen_size == (1600, 900):
            raise RuntimeError("Screen size is not 1600x900")
        return ImageReference(1225, 20, 100, 20, "references/minigame_retry_cross_button.png")

    @property
    def image_classifier(self):
        if not hasattr(self, "_image_classifier"):
            self._image_classifier = ImageClassifier("references/sprites/raw", save_predictions=True)
        return self._image_classifier

    def _get_current_cross_button(self):
        return self.find_reference("references/cross_button.png")

    def press_cross_button(self):
        if self.has_reference(self._minigame_quit_reference):
            self.tap(self._minigame_quit_reference)
            time.sleep(1)
            logger.debug("Quitted minigame")
            return True

        if self.has_reference(self._minigame_retry_cross_reference):
            self.tap(self._minigame_retry_cross_reference)
            time.sleep(1)
            logger.debug("Deleted minigame retry prompt")
            return True

        if self.has_reference(self._minigame_screen_cross_reference):
            self.tap(self._minigame_screen_cross_reference)
            time.sleep(1)
            logger.debug("Closed minigame screen")
            return True

        cross_button = self._get_current_cross_button()
        if cross_button is not None:
            self.tap(cross_button)
            return True
        return False

    def open_minigames_screen(self):
        logger.debug("Trying to open minigame screen")

        if self.has_reference(self._minigame_screen_reference):
            logger.debug("Minigame screen already open")
            return True

            logger.info("Minigame not available")
            return False

        self.tap(self._minigame_reference)
        logger.info("Opened minigame screen")
        return True

    def start_minigame(self):
        logger.debug("Trying to start minigame")
        if self.has_reference(self._minigame_start_reference):
            self.tap(self._minigame_start_reference)
            logger.info("Started minigame screen")
            return True
        logger.info("Minigame start button not found")
        return False

    def solve_minigame(self):
        logger.debug("Solving minigame")
        try:
            grid = detect_grid(self.screencap(), self.image_classifier)
        except InvalidGrid as e:
            logger.error(f"Invalid grid: {e}")
            return False
        logger.debug("Detected grid")
        logger.debug(show_grid(grid))

        solution = solve_grid(grid)
        if solution is None:
            logger.error("No solution found")
            return False

        logger.debug("Found solution")
        for p0, p1, p2, p3 in solution:
            logger.debug(f"({p0}, {p1}) -> ({p2}, {p3})")

        if solution is not None:
            for p0, p1, p2, p3 in solution:
                self.tap(get_grid_reference(p0, p1, 20))
                self.tap(get_grid_reference(p2, p3, 20))

        time.sleep(2)
        if self.has_reference(self._minigame_ingame_reference):
            logger.error("Failed to solve minigame")
            if self.has_reference(self._minigame_ingame_reference) and self.has_reference(self._minigame_ingame_cross_reference):
                self.tap(self._minigame_ingame_cross_reference)
                logger.info("Cross button pressed")
                time.sleep(1)
            return False

        logger.info("Solved minigame")
        return True

def run(em: SummikoEmulator):
    while True:
        if em.has_reference(em._minigame_reference) and em.open_minigames_screen():
            logger.info("Minigame screen opened")
            time.sleep(1)
            continue

        if em.has_reference(em._minigame_screen_reference) and em.has_reference(em._minigame_start_reference) and em.start_minigame():
            logger.info("Minigame started")
            time.sleep(1)
            continue

        if em.has_reference(em._minigame_screen_reference) and not em.has_reference(em._minigame_start_reference):
            # Minigame not available yet
            logger.info("Minigame not available yet")
            time.sleep(1)
            continue

        if em.has_reference(em._minigame_ingame_reference):
            logger.info("Minigame in progress")
            if em.solve_minigame():
                logger.info("Minigame solved")
                time.sleep(1)
                continue

            logger.info("Failed to solve minigame")

        if em.press_cross_button():
            logger.info("Game fixed")
            time.sleep(1)
            continue

        logger.info("Nothing to do")
        time.sleep(3)

def main():
    em = SummikoEmulator("localhost:21503")
    try:
        run(em)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        logger.exception(e)

if __name__ == "__main__":
    main()
