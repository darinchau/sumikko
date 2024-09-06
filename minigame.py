from emulator import Emulator, ImageReference, save_screenshot, Image
import numpy as np
from numpy.typing import NDArray
import os
from contextlib import contextmanager
from datetime import datetime

def get_grid_reference(i: int, j: int):
    """Get the reference for a grid cell."""
    return ImageReference(
        75 + 110 * i + 55,
        55,
        120 + 110 * j + 55,
        55,
        f"references/grid_{i}_{j}.png")

def create_grid_reference(em: Emulator):
    """Run once at some point in time to get all the sprites for the grid."""
    screenshot = em.screencap()
    for i in range(11):
        for j in range(6):
            imref = get_grid_reference(i, j)
            print(f"Making refernece {i}, {j} at {imref.x_center}, {imref.y_center}")
            em.make_reference(imref, screenshot)

def take_grid_screenshot(screenshot: Image, reference_path: str = "./references/sprites", save: bool = False):
    """Detect the grid from the screenshot. Returns a 11x6 grid. 0 is empty, 1-15 are the sprites, -1 is the box."""
    assert screenshot.shape == (900, 1600, 3)
    dims = get_grid_reference(0, 0).extract(screenshot).shape
    grid_screenshot = np.zeros((11, 6, dims[0], dims[1], dims[2]), dtype=np.float32)
    for i in range(11):
        for j in range(6):
            sprite = get_grid_reference(i, j).extract(screenshot)
            save_screenshot(sprite, os.path.join(reference_path, "raw"), image_name=f"sprite_{i}_{j}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
            grid_screenshot[i, j] = sprite
    return grid_screenshot

def detect_grid(screenshot: Image):
    """Detect the grid from the screenshot. Returns a 11x6 grid. 0 is empty, >0 are the sprites, -1 is the box."""
    raise NotImplementedError("This function is not implemented yet.")
    sprites_path = [os.path.join(reference_path, f) for f in os.listdir(reference_path) if f.startswith("sprite") and f.endswith(".png")]
    sprites_path.append(os.path.join(reference_path, "box.png"))
    nsprites = len(sprites_path)

    sprites = np.zeros((nsprites, 40, 40, 3))
    for i, pth in enumerate(sprites_path):
        sprites[i] = Emulator.read_image(pth)

    dists = np.zeros((11, 6, nsprites))
    for i in range(11):
        for j in range(6):
            roi = get_grid_reference(i, j).extract(screenshot)
            for k in range(nsprites):
                dists[i, j, k] = np.sum(np.abs(roi - sprites[k])) / 40 / 40 / 3

def is_valid_grid(grid: NDArray[np.int64]) -> bool:
    nsprites = grid.max()
    counts = [0] * (nsprites + 1)
    for i in range(11):
        for j in range(6):
            counts[grid[i, j]] += 1

    npairs = 0
    for i in range(1, nsprites):
        if not counts[i] in (0, 2, 4):
            # print(f"Invalid count for sprite {i}: {counts[i]}")
            return False
        npairs += counts[i] // 2

    return True

def show_grid(grid: NDArray[np.int64]):
    """Print the grid to the console."""
    s = "\n"
    for i in range(6):
        for j in range(11):
            s += str(grid[j, i]).ljust(3)
        s += "\n"
    return s

@contextmanager
def lock_grid(grid: NDArray[np.int64]):
    grid.flags.writeable = False
    yield grid
    grid.flags.writeable = True

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
    if (i, j) in visited and visited[(i, j)] <= ntwists:
        return
    visited[(i, j)] = ntwists

    # If the cell is not empty, then we can't go through it
    if 0 <= i < grid.shape[0] and 0 <= j < grid.shape[1] and grid[i, j] != 0:
        return

    at_left_boundary = i == -1 and (0 <= j < grid.shape[1])
    at_right_boundary = i == grid.shape[0] and (0 <= j < grid.shape[1])
    at_top_boundary = j == -1 and (0 <= i <= grid.shape[0])
    at_bottom_boundary = j == grid.shape[1] and (0 < i < grid.shape[0])

    can_go_left = i == 0 or at_top_boundary or at_bottom_boundary or (i > 0 and 0 <= j < grid.shape[1])
    can_go_right = i == grid.shape[0] - 1 or at_top_boundary or at_bottom_boundary or (i < grid.shape[0] - 1 and 0 <= j < grid.shape[1])
    can_go_up = j == 0 or at_left_boundary or at_right_boundary or (j > 0 and 0 <= i < grid.shape[0])
    can_go_down = j == grid.shape[1] - 1 or at_left_boundary or at_right_boundary or (j < grid.shape[1] - 1 and 0 <= i < grid.shape[0])

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
    if i == 0 or (i > 0 and grid[i - 1, j] == 0):
        _find_all_paths(grid, i - 1, j, "L", 0, visited)
    if i == grid.shape[0] - 1 or (i < grid.shape[0] - 1 and grid[i + 1, j] == 0):
        _find_all_paths(grid, i + 1, j, "R", 0, visited)
    if j == 0 or (j > 0 and grid[i, j - 1] == 0):
        _find_all_paths(grid, i, j - 1, "U", 0, visited)
    if j == grid.shape[1] - 1 or (j < grid.shape[1] - 1 and grid[i, j + 1] == 0):
        _find_all_paths(grid, i, j + 1, "D", 0, visited)
    paths = []
    for x in range(11):
        for y in range(6):
            if (x, y) in visited and (i, j) != (x, y) and grid[x, y] == grid[i, j] and 0 <= x < grid.shape[0] and 0 <= y < grid.shape[1]:
                paths.append((x, y))
    return paths

def _solve_grid(grid: NDArray[np.int64]) -> list[tuple[int, int, int, int]] | None:
    is_solved = np.all(grid <= 0)
    if is_solved:
        return []

    available_paths: list[tuple[int, int, int, int]] = []
    with lock_grid(grid):
        for i in range(11):
            for j in range(6):
                if grid[i, j] <= 0:
                    continue
                paths = find_all_paths(grid, i, j)
                for path in paths:
                    available_paths.append((i, j, path[0], path[1]))

    if not available_paths:
        return None

    for p0, p1, p2, p3 in available_paths:
        assert grid[p0, p1] == grid[p2, p3] and grid[p0, p1] > 0
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
