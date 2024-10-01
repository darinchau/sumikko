from __future__ import annotations
import os
from emulator import Emulator, ImageReference, save_screenshot
import numpy as np
from numpy.typing import NDArray
import logging
from minigame import detect_grid, get_grid_reference, show_grid, solve_grid, InvalidGrid, save_screenshot
import time
import numba
from model import ImageClassifier
from abc import ABC, abstractmethod
import tempfile
from contextlib import contextmanager
import easyocr

CROSS_BUTTON_DIST_THRESHOLD = 0.03

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class Action(ABC):
    @property
    def logger(self):
        return logger

    def validate_emulator(self, emulator: Emulator):
        if not emulator.screen_size == (1600, 900):
            raise RuntimeError("Screen size is not 1600x900")

    @abstractmethod
    def perform(self, emulator: Emulator) -> list[Action]:
        """Perform the action. You can assume the condition is met. Returns a list of possible actions after performing the action."""
        pass

    @abstractmethod
    def condition(self, emulator: Emulator) -> bool:
        """Check if the action can be performed. Returns True if the action can be performed."""
        pass

    def __repr__(self):
        return self.__class__.__name__

class WaitAction(Action):
    def __init__(self, duration: float):
        self.duration = duration

    def perform(self, emulator: Emulator) -> list[Action]:
        self.logger.debug(f"Waiting for {self.duration} seconds")
        time.sleep(self.duration)
        return []

    def condition(self, emulator: Emulator) -> bool:
        return True

class OpenMinigameAction(Action):
    """Open the minigame screen."""
    @property
    def minigame_screen_reference(self) -> ImageReference:
        """The reference image for the minigame screen."""
        return ImageReference(792, 100, 99, 20, "references/minigame_screen_reference.png")

    @property
    def minigame_reference(self) -> ImageReference:
        """The minigame icon at the bottom right on the main screen"""
        return ImageReference(1060, 40, 810, 50, "references/minigame_reference.png")

    def perform(self, emulator: Emulator) -> list[Action]:
        self.validate_emulator(emulator)
        self.logger.debug("Trying to open minigame screen")

        if emulator.has_reference(self.minigame_screen_reference):
            self.logger.debug("Minigame screen already open")
        else:
            emulator.tap(self.minigame_reference)
            self.logger.info("Opened minigame screen")
        return [
            StartMinigameAction()
        ]

    def condition(self, emulator: Emulator) -> bool:
        return emulator.has_reference(self.minigame_reference)

class StartMinigameAction(Action):
    """Indicates we are on the mini game screen and we want to start the game."""
    @property
    def minigame_screen_reference(self) -> ImageReference:
        """The reference image for the minigame screen."""
        return ImageReference(792, 100, 99, 20, "references/minigame_screen_reference.png")

    @property
    def minigame_start_reference(self):
        """The minigame start button"""
        return ImageReference(790, 150, 710, 20, "references/minigame_start_reference.png")

    def perform(self, emulator: Emulator) -> list[Action]:
        self.logger.debug("Trying to start minigame")
        if emulator.has_reference(self.minigame_start_reference):
            emulator.tap(self.minigame_start_reference)
            self.logger.info("Started minigame screen")
            return [
                PlayMinigameAction()
            ]
        self.logger.debug("Minigame not available yet")
        return [
            OpenMinigameAction()
        ]

    def condition(self, emulator: Emulator) -> bool:
        return emulator.has_reference(self.minigame_screen_reference)

class PlayMinigameAction(Action):
    @property
    def minigame_ingame_reference(self):
        """A reference which indicates we are in the minigame."""
        return ImageReference(1380, 40, 100, 40, "references/minigame_ingame_reference.png")

    @property
    def image_classifier(self):
        if not hasattr(self, "_image_classifier"):
            self._image_classifier = ImageClassifier("references/sprites/raw", save_predictions=False)
        return self._image_classifier

    def perform(self, emulator: Emulator) -> list[Action]:
        self.logger.debug("Solving minigame")
        try:
            grid = detect_grid(emulator.screencap(), self.image_classifier)
        except InvalidGrid as e:
            self.logger.error(f"Invalid grid: {e}")
            return [PlayMinigameAction()]

        self.logger.debug("Detected grid")
        self.logger.debug(show_grid(grid))

        solution = solve_grid(grid)
        if solution is None:
            self.logger.error("No solution found")
            return [CrossButtonAction()]

        self.logger.debug("Found solution")
        for p0, p1, p2, p3 in solution:
            self.logger.debug(f"({p0}, {p1}) -> ({p2}, {p3})")

        for p0, p1, p2, p3 in solution:
            emulator.tap(get_grid_reference(p0, p1, 20))
            emulator.tap(get_grid_reference(p2, p3, 20))

        time.sleep(2)

        # Retry by recursively calling solve_minigame
        if emulator.has_reference(self.minigame_ingame_reference):
            return [
                PlayMinigameAction()
            ]

        self.logger.info("Solved minigame")
        return [WaitAction(1)]

    def condition(self, emulator: Emulator) -> bool:
        return emulator.has_reference(self.minigame_ingame_reference)

class MinigameQuitAction(Action):
    @property
    def minigame_quit_reference(self):
        return ImageReference(1000, 100, 600, 20, "references/minigame_quit_button.png")

    def perform(self, emulator: Emulator) -> list[Action]:
        if emulator.has_reference(self.minigame_quit_reference):
            emulator.tap(self.minigame_quit_reference)
            self.logger.debug("Quitted minigame")
            return [WaitAction(1)]
        return []

    def condition(self, emulator: Emulator) -> bool:
        return emulator.has_reference(self.minigame_quit_reference)

class CrossButtonAction(Action):
    @property
    def cross_button_base_color(self) -> NDArray[np.float32]:
        return np.array([255, 138, 132], dtype=np.float32) / 255

    @property
    def minigame_quit_reference(self):
        return ImageReference(1000, 100, 600, 20, "references/minigame_quit_button.png")

    def perform(self, emulator: Emulator) -> list[Action]:
        img = emulator.screencap()
        cross_button = find_cross_button_reference(img, self.cross_button_base_color)
        if cross_button is None:
            self.logger.debug("Cross button not found")
            return []

        emulator.tap(cross_button)
        self.logger.info(f"Pressed cross button at {cross_button.x_center}, {cross_button.y_center}")
        return [WaitAction(1)]

    def condition(self, emulator: Emulator) -> bool:
        img = emulator.screencap()
        maybe_has_cross_button = np.count_nonzero(np.linalg.norm(img - self.cross_button_base_color, axis=-1) < CROSS_BUTTON_DIST_THRESHOLD) > 0
        return maybe_has_cross_button

class FailedToReadDownloadDataAction(Action):
    @property
    def download_data_reference(self):
        return ImageReference(803, 200, 400, 10, "references/failed download.png")

    @property
    def download_data_ok_reference(self):
        return ImageReference(800, 150, 570, 20, "references/failed download ok.png")

    def perform(self, emulator: Emulator) -> list[Action]:
        if emulator.has_reference(self.download_data_reference) and emulator.has_reference(self.download_data_ok_reference):
            emulator.tap(self.download_data_ok_reference)
            self.logger.info("Fixed failed download")
            return [WaitAction(1)]
        return []

    def condition(self, emulator: Emulator) -> bool:
        return emulator.has_reference(self.download_data_reference)

class TitleScreenAction(Action):
    @property
    def title_screen_settings_reference(self):
        return ImageReference(1480, 40, 815, 20, "references/title screen setting.png")

    @property
    def title_screen_reference(self):
        return ImageReference(800, 300, 450, 200, "references/title_screen_clickable.png")

    def perform(self, emulator: Emulator) -> list[Action]:
        if emulator.has_reference(self.title_screen_settings_reference):
            emulator.tap(self.title_screen_reference)
            self.logger.info("Tapped title screen")
            return [WaitAction(1)]
        return []

    def condition(self, emulator: Emulator) -> bool:
        return emulator.has_reference(self.title_screen_settings_reference)

class WordButtonAction(Action):
    def __init__(self, word: str):
        self.word = word

    @contextmanager
    def get_screenshot(self, emulator: Emulator):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = save_screenshot(emulator.screencap(), tmpdir)
            yield path

    def perform(self, emulator: Emulator) -> list[Action]:
        with self.get_screenshot(emulator) as path:
            word_boxes = find_word_in_image(path, self.word)
        if len(word_boxes) > 0:
            x1, y1, x2, y2 = word_boxes[0]
            imgref = ImageReference((x1 + x2) // 2, x2 - x1 + 1, (y1 + y2) // 2, y2 - y1 + 1, "N/A")
            emulator.tap(imgref)
            self.logger.info(f"Pressed {self.word} button")
            return [WaitAction(1)]
        return []

    def condition(self, emulator: Emulator) -> bool:
        with self.get_screenshot(emulator) as path:
            word_boxes = find_word_in_image(path, self.word)
        return len(word_boxes) > 0

    def __repr__(self):
        return f"{self.word.title()}ButtonAction"

@numba.jit(nopython=True)
def _find_cross_button_reference(img: NDArray[np.float32], base_color: NDArray[np.float32]) -> NDArray[np.int64]:
    """Find the cross button reference. Returns a 4-tuple in the form of (left, right, top, bottom)."""
    cross_button_color_dists = np.abs(img - base_color)
    cross_button_color_dists = np.sum(cross_button_color_dists, axis=-1)
    cross_button_map = cross_button_color_dists < CROSS_BUTTON_DIST_THRESHOLD
    dp = np.zeros(cross_button_map.shape, dtype=np.int64)
    dp[0] = cross_button_map[0]
    for i in range(1, cross_button_map.shape[0]):
        for j in range(cross_button_map.shape[1]):
            if cross_button_map[i, j]:
                dp[i, j] = dp[i-1, j] + 1
    max_area = 0
    max_left_bound = 0
    max_right_bound = 0
    max_top_bound = 0
    max_bottom_bound = 0
    for i in range(cross_button_map.shape[0]):
        stack = np.zeros((cross_button_map.shape[1] + 1,), dtype=np.int64)
        stack_idx = 0
        for j in range(cross_button_map.shape[1]):
            while stack_idx > 0 and dp[i, j] < dp[i, stack[stack_idx - 1]]:
                height = dp[i, stack[stack_idx - 1]]
                stack_idx -= 1
                width = j if stack_idx == 0 else j - stack[stack_idx - 1] - 1
                if height * width > max_area:
                    max_area = height * width
                    max_left_bound = j - width + 1
                    max_right_bound = j
                    max_top_bound = i - height + 1
                    max_bottom_bound = i
            stack[stack_idx] = j
            stack_idx += 1

        while stack_idx > 0:
            height = dp[i, stack[stack_idx - 1]]
            stack_idx -= 1
            width = cross_button_map.shape[1] if stack_idx == 0 else cross_button_map.shape[1] - stack[stack_idx - 1] - 1
            if height * width > max_area:
                max_area = height * width
                max_left_bound = j - width + 1
                max_right_bound = cross_button_map.shape[1] - 1
                max_top_bound = i - height + 1
                max_bottom_bound = i

    if max_area < 5:
        return np.array([-1, -1, -1, -1])
    return np.array([max_left_bound, max_right_bound, max_top_bound, max_bottom_bound])

def find_cross_button_reference(img: NDArray[np.float32], base_color: NDArray[np.float32]) -> ImageReference | None:
    bounding_box = _find_cross_button_reference(img, base_color)
    l, r, t, b = bounding_box
    if l == r == t == b == -1:
        return None
    return ImageReference((l + r) // 2, r - l + 1, (t + b) // 2, b - t + 1, "N/A")

def find_word_in_image(image_path, word):
    reader = easyocr.Reader(['en'])

    results = reader.readtext(image_path, detail=1)

    word_boxes = []

    for (bbox, text, prob) in results:
        if word.lower() == text.lower():
            x1, y1 = int(bbox[0][0]), int(bbox[0][1])
            x2, y2 = int(bbox[2][0]), int(bbox[2][1])
            word_boxes.append((x1, y1, x2, y2))

    return word_boxes
