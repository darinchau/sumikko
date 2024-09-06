import subprocess
import logging
import tempfile
import matplotlib.image as mpimg
from numpy.typing import NDArray
import numpy as np
import os
from datetime import datetime
from dataclasses import dataclass
import numba

logger = logging.getLogger(__name__)

Image = NDArray[np.float32]

@dataclass(frozen=True)
class ImageReference:
    """Defines an image reference."""
    x_center: int
    x_range: int
    y_center: int
    y_range: int
    reference_path: str

    def __post_init__(self):
        if self.available:
            image = mpimg.imread(self.reference_path)
            assert image.shape[0] == 2 * self.y_range
            assert image.shape[1] == 2 * self.x_range

    @property
    def available(self) -> bool:
        """Check if the reference image is available."""
        return os.path.exists(self.reference_path)

    def __repr__(self) -> str:
        return f"ImageReference({self.reference_path})"

    def extract(self, screenshot: Image) -> Image:
        """Extract the image from the screenshot."""
        return screenshot[self.y_center - self.y_range: self.y_center + self.y_range, self.x_center - self.x_range: self.x_center + self.x_range]

class Emulator:
    def __init__(self, ip: str):
        self.frame_thread = None
        self.video_thread = None
        self.frame = None
        self.ip = ip

        self._connect_to_device()

    def _run_adb_command(self, command: list[str]):
        try:
            output = subprocess.check_output(["adb"] + command)
        except Exception as e:
            logger.error(f"Error running command: {str(e)}")
            raise RuntimeError("Error running command") from e

        return output

    def _connect_to_device(self):
        """Run the necessary adb command to connect to the device."""
        try:
            self._run_adb_command(["connect", self.ip])
        except Exception as e:
            logger.error(f"Error connecting to device: {str(e)}")
            raise RuntimeError("Could not connect to the device") from e

        logger.info(f"Successfully connected to device at {self.ip}")

    def _get_screen_size(self):
        """Get the screen size of the device."""
        try:
            output = self._run_adb_command(["shell", "wm", "size"])
        except Exception as e:
            logger.error(f"Error getting screen size: {str(e)}")
            raise RuntimeError("Error getting screen size") from e

        output = output.decode("utf-8").strip()
        size = output.split("Physical size: ")[1].split("\n")[0]
        width, height = size.split("x")
        return int(width), int(height)

    @staticmethod
    def read_image(path: str) -> Image:
        """Read an image from the path."""
        return mpimg.imread(path)[:, :, :3].astype(np.float32)

    @property
    def screen_size(self):
        """Get the screen size of the device."""
        if not hasattr(self, "_screen_size"):
            self._screen_size = self._get_screen_size()
        return self._screen_size

    def screencap(self, save: bool = True):
        """Take a screenshot of the device. The screenshot is returned as a numpy array with shape (height, width, 3)."""
        try:
            self._run_adb_command(["shell", "screencap", "/sdcard/screen.png"])

        except Exception as e:
            logger.error(f"Error taking screenshot: {str(e)}")
            raise RuntimeError("Error taking screenshot") from e

        with tempfile.NamedTemporaryFile(suffix=".png") as f:
            try:
                self._run_adb_command(["pull", "/sdcard/screen.png", f.name])
                frame = self.read_image(f.name)
            except Exception as e:
                logger.error(f"Error pulling screenshot: {str(e)}")
                raise RuntimeError("Error pulling screenshot") from e

        assert frame.shape[0] == self.screen_size[1]
        assert frame.shape[1] == self.screen_size[0]
        assert frame.shape[2] == 3
        logger.debug("Successfully took screenshot")

        if save:
            save_screenshot(frame, "./screenshot")
        return frame

    def tap(self, x: int | ImageReference, y: int | None = None):
        """Tap the screen at the specified coordinates."""
        assert (isinstance(x, int) and isinstance(y, int)) or (isinstance(x, ImageReference) and y is None)
        if isinstance(x, ImageReference):
            x, y = x.x_center, x.y_center
        try:
            self._run_adb_command(["shell", "input", "tap", str(x), str(y)])
        except Exception as e:
            logger.error(f"Error tapping screen: {str(e)}")
            raise RuntimeError("Error tapping screen") from e

        logger.debug(f"Successfully tapped screen at ({x}, {y})")

    def make_reference(self, ref: ImageReference, screenshot: Image | None = None):
        """Save a reference image of the screen."""
        screenshot = screenshot if screenshot is not None else self.screencap(save=False)
        reference = ref.extract(screenshot)
        mpimg.imsave(ref.reference_path, reference)

    def has_reference(self, ref: ImageReference, screenshot: Image | None = None, diff_threshold: float = 0.01) -> bool:
        """Check if the screen has a reference image."""
        logger.debug(f"Checking reference: {ref}")
        if not ref.available:
            logger.error(f"Reference image: {ref} not available")
            return False
        screenshot = screenshot if screenshot is not None else self.screencap()
        screenshot = ref.extract(screenshot)
        reference = self.read_image(ref.reference_path)
        return compare_image(screenshot, reference)  < diff_threshold

    def find_reference(self, ref_path: str, screenshot: Image | None = None, diff_threshold: float = 0.01) -> ImageReference | None:
        ref = self.read_image(ref_path)
        screenshot = screenshot if screenshot is not None else self.screencap()
        pixels = find_reference(ref, screenshot, diff_threshold)
        if np.all(pixels == -1):
            return None
        x = pixels[1].item()
        y = pixels[0].item()
        x_range = ref.shape[1] // 2
        y_range = ref.shape[0] // 2
        ref = ImageReference(x + x_range, x_range, y + y_range, y_range, ref_path)
        return ref

def save_screenshot(array: Image, screenshot_dir: str = "./screenshot", image_name: str | None = None):
    if not os.path.exists(screenshot_dir):
        os.makedirs(screenshot_dir)

    if image_name is None:
        image_name = f"screenshot_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    else:
        image_name = f"{image_name}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    screenshot_path = os.path.join(screenshot_dir, f"{image_name}.png")
    mpimg.imsave(screenshot_path, array)

def compare_image(image1: Image, reference: str | Image) -> float:
    """Compare two images and return the difference."""
    if isinstance(reference, str):
        image2 = Emulator.read_image(reference)
    else:
        image2 = reference
    if image1.shape != image2.shape:
        return False
    if image1.dtype != image2.dtype:
        return False
    diff = np.sum(np.abs(image1 - image2)) / np.prod(image1.shape)
    return diff

@numba.jit(nopython=True)
def find_reference(ref: Image, image: Image, diff_threshold: float = 0.01):
    """Find a reference image on the screen."""
    npixels = ref.size
    for i in range(image.shape[0] - ref.shape[0]):
        for j in range(image.shape[1] - ref.shape[1]):
            roi = image[i: i + ref.shape[0], j: j + ref.shape[1]]
            dist = np.sum(np.abs(roi - ref)) / npixels
            if dist < diff_threshold:
                return np.array([i, j])
    return np.array([-1, -1])