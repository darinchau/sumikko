from emulator import Emulator, ImageReference, save_screenshot, Image
import numpy as np
from numpy.typing import NDArray
import os
from contextlib import contextmanager
from datetime import datetime
from model import ImageClassifier
from scipy.spatial.distance import squareform, pdist

GRID_SIZE = (11, 6)

class InvalidGrid(ValueError):
    pass

def get_grid_reference(i: int, j: int, radius: int):
    """Get the reference for a grid cell."""
    return ImageReference(
        75 + 110 * i + 55,
        radius,
        120 + 110 * j + 55,
        radius,
        f"references/grid_{i}_{j}.png")

def take_grid_screenshot(screenshot: Image, reference_path: str = "./references/sprites", radius: int = 55, save: bool = False):
    """Used liberally to take screenshots of the grid and save the sprites."""
    assert screenshot.shape == (900, 1600, 3)
    grid_screenshot = np.zeros(GRID_SIZE + (2 * radius, 2 * radius, 3), dtype=np.float32)
    for i in range(GRID_SIZE[0]):
        for j in range(GRID_SIZE[1]):
            sprite = get_grid_reference(i, j, radius).extract(screenshot)
            if save:
                save_screenshot(sprite, os.path.join(reference_path, "raw"), image_name=f"sprite_{i}_{j}")
            grid_screenshot[i, j] = sprite
    return grid_screenshot

def detect_grid(screenshot: Image, classifier: ImageClassifier, same_sprite_threshold: float = 0.4):
    """Detect the grid from the screenshot. Returns a 11x6 grid. 0 is empty, >0 are the sprites, -1 is the box."""
    grid_screenshot = take_grid_screenshot(screenshot, radius = 55, save=False)
    # If all black or all white, then it's an invalid grid
    if np.average(grid_screenshot) < 0.01:
        raise InvalidGrid("All black images")
    if np.average(grid_screenshot) > 0.99:
        raise InvalidGrid("All white images")

    # Construct the grid by using pairwise image distances to try and get the best match
    predictions = classifier.predict_grid_screenshot(grid_screenshot)
    grid = np.zeros(GRID_SIZE, dtype=np.int64)
    grid[predictions == 1] = -1

    # Retake the screenshot and save the sprites
    grid_screenshot = take_grid_screenshot(screenshot, radius = 20, save=False).reshape(-1, 40 * 40 * 3)
    grid_euclidean_dist = squareform(pdist(grid_screenshot, metric="euclidean"))
    sorted_distance_idx = np.argsort(grid_euclidean_dist, axis=1)
    sorted_distances = np.sort(grid_euclidean_dist, axis=1)

    # Group the sprites together by distance
    # Strategy is to find all the sprites that are really close together, put them together somewhere,
    # and then use a union-find algorithm to group them together
    sprite_groupings = set()
    for i in range(GRID_SIZE[0]):
        for j in range(GRID_SIZE[1]):
            ref = i * GRID_SIZE[1] + j
            if predictions[i, j] in (0, 1):
                continue

            for k in range(1, 4):
                if sorted_distances[ref, k] < same_sprite_threshold:
                    a, b = ref, sorted_distance_idx[ref, k].item()
                    if a == b:
                        continue
                    if a > b:
                        a, b = b, a
                    sprite_groupings.add((a, b))

    sprite_groupings = sorted(sprite_groupings)

    # Union-find algorithm
    sprite_idxs = {}
    sprite_idx = 1
    for g1, g2 in sprite_groupings:
        if g1 not in sprite_idxs and g2 not in sprite_idxs:
            sprite_idxs[g1] = sprite_idxs[g2] = sprite_idx
            sprite_idx += 1
        elif g1 in sprite_idxs and g2 not in sprite_idxs:
            sprite_idxs[g2] = sprite_idxs[g1]
        elif g2 in sprite_idxs and g1 not in sprite_idxs:
            sprite_idxs[g1] = sprite_idxs[g2]
        else:
            a, b = sprite_idxs[g1], sprite_idxs[g2]
            for i in sprite_idxs:
                if sprite_idxs[i] == b:
                    sprite_idxs[i] = a

    # Make the grid by using the sprite groupings
    for i in range(GRID_SIZE[0]):
        for j in range(GRID_SIZE[1]):
            ref = i * GRID_SIZE[1] + j
            if predictions[i, j] == 2:
                grid[i, j] = sprite_idxs[ref]

    # Sanity check section: check if the grid is valid
    nsprites = grid.max()
    counts = [0] * (nsprites + 1)
    for i in range(GRID_SIZE[0]):
        for j in range(GRID_SIZE[1]):
            counts[grid[i, j]] += 1

    npairs = 0
    for i in range(1, nsprites):
        if not counts[i] in (0, 2, 4):
            InvalidGrid(f"Invalid count for sprite {i}: {counts[i]}")
        npairs += counts[i] // 2
    return grid

def show_grid(grid: NDArray[np.int64]):
    assert grid.shape == GRID_SIZE
    s = "\n"
    for i in range(GRID_SIZE[1]):
        for j in range(GRID_SIZE[0]):
            s += str(grid[j, i]).ljust(3)
        s += "\n"
    return s

# A recursive implementation of the backtracking algorithm to find all available paths
# L means i -> i-1
# R means i -> i+1
# U means j -> j-1
# D means j -> j+1
# visited is a dictionary of (i, j) -> lowest number of twists to reach that cell
def _find_all_paths(grid: np.ndarray, i: int, j: int, current_direction: str, ntwists: int, visited: dict[tuple[int, int], int]):
    assert current_direction in ("U", "D", "L", "R")
    if ntwists > 2:
        return
    if (i, j) in visited and visited[(i, j)] < ntwists:
        return
    visited[(i, j)] = ntwists

    # If the cell is not empty, then we can't go through it
    if 0 <= i < grid.shape[0] and 0 <= j < grid.shape[1] and grid[i, j] != 0:
        return

    at_left_boundary = i == -1 and (0 <= j < grid.shape[1])
    at_right_boundary = i == grid.shape[0] and (0 <= j < grid.shape[1])
    at_top_boundary = j == -1 and (0 <= i <= grid.shape[0])
    at_bottom_boundary = j == grid.shape[1] and (0 < i < grid.shape[0])

    can_go_left = current_direction != "R" and i == 0 or at_top_boundary or at_bottom_boundary or (i > 0 and 0 <= j < grid.shape[1])
    can_go_right = current_direction != "L" and i == grid.shape[0] - 1 or at_top_boundary or at_bottom_boundary or (i < grid.shape[0] - 1 and 0 <= j < grid.shape[1])
    can_go_up = current_direction != "U" and j == 0 or at_left_boundary or at_right_boundary or (j > 0 and 0 <= i < grid.shape[0])
    can_go_down = current_direction != "D" and j == grid.shape[1] - 1 or at_left_boundary or at_right_boundary or (j < grid.shape[1] - 1 and 0 <= i < grid.shape[0])

    if can_go_left:
        _find_all_paths(grid, i - 1, j, "L", ntwists if current_direction=="L" else ntwists+1, visited)

    if can_go_right:
        _find_all_paths(grid, i + 1, j, "R", ntwists if current_direction=="R" else ntwists+1, visited)

    if can_go_up:
        _find_all_paths(grid, i, j - 1, "U", ntwists if current_direction=="U" else ntwists+1, visited)

    if can_go_down:
        _find_all_paths(grid, i, j + 1, "D", ntwists if current_direction=="D" else ntwists+1, visited)

def find_all_paths(grid: np.ndarray, i: int, j: int):
    """Find all paths from (i, j) to other cells of the same value. Do some checks and makes the initial call to the recursive DFS function."""
    assert 0 <= i < grid.shape[0]
    assert 0 <= j < grid.shape[1]
    visited = {}

    if i == 0 or (i > 0 and grid[i - 1, j] >= 0):
        _find_all_paths(grid, i - 1, j, "L", 0, visited)
    if i == grid.shape[0] - 1 or (i < grid.shape[0] - 1 and grid[i + 1, j] >= 0):
        _find_all_paths(grid, i + 1, j, "R", 0, visited)
    if j == 0 or (j > 0 and grid[i, j - 1] >= 0):
        _find_all_paths(grid, i, j - 1, "U", 0, visited)
    if j == grid.shape[1] - 1 or (j < grid.shape[1] - 1 and grid[i, j + 1] >= 0):
        _find_all_paths(grid, i, j + 1, "D", 0, visited)

    paths = []
    for x in range(GRID_SIZE[0]):
        for y in range(GRID_SIZE[1]):
            if (x, y) in visited and (i, j) != (x, y) and grid[x, y] == grid[i, j] and 0 <= x < grid.shape[0] and 0 <= y < grid.shape[1]:
                paths.append((x, y))
    return paths

def _solve_grid(grid: NDArray[np.int64]) -> list[tuple[int, int, int, int]] | None:
    is_solved = np.all(grid <= 0)
    if is_solved:
        return []

    available_paths: list[tuple[int, int, int, int]] = []
    for i in range(11):
        for j in range(6):
            if grid[i, j] <= 0:
                continue
            paths = find_all_paths(grid, i, j)
            for path in paths:
                if (path[0], path[1], i, j) in available_paths or (i, j, path[0], path[1]) in available_paths:
                    continue
                available_paths.append((i, j, path[0], path[1]))

    if not available_paths:
        return None

    available_paths = sorted(available_paths, key=lambda x: (grid[x[0], x[1]], (x[0], x[1])))

    sprites, counts = np.unique(grid, return_counts=True)

    # Sanity Check
    for p0, p1, p2, p3 in available_paths:
        assert grid[p0, p1] == grid[p2, p3] and grid[p0, p1] > 0

    # Shortcut the computation by removing sprites that only have one pair left
    shortcut_paths = []
    for p0, p1, p2, p3 in available_paths:
        for i, c in enumerate(counts):
            if c == 2 and sprites[i] == grid[p0, p1] == grid[p2, p3]:
                grid[p0, p1] = grid[p2, p3] = 0
                shortcut_paths.append((p0, p1, p2, p3))

    if shortcut_paths:
        solution = _solve_grid(grid)
        if solution is not None:
            return shortcut_paths + solution
        return None

    # Do backtracking normally
    for p0, p1, p2, p3 in available_paths:
        orig_value = grid[p0, p1]
        grid[p0, p1] = grid[p2, p3] = 0
        solution = _solve_grid(grid)
        grid[p0, p1] = grid[p2, p3] = orig_value
        if solution is not None:
            return [(p0, p1, p2, p3)] + solution

    return None

def solve_grid(grid: NDArray[np.int64]) -> list[tuple[int, int, int, int]] | None:
    """Solve the grid and return the list of paths. None if no solution is found. Makes the initial call to the recursives backtracking algorithm."""
    assert grid.shape == (11, 6)
    return _solve_grid(grid)
