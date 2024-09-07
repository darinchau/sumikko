# Use a KNN model to predict the class of a given image
import os
from emulator import Image, ImageReference, Emulator, save_screenshot
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score

class ImageClassifier:
    def __init__(self, ref_path: str, save_predictions: bool = False):
        self.ref_path = ref_path
        self.save_predictions = save_predictions
        self.x, self.y = load_images(ref_path)
        self.knn = KNeighborsClassifier(n_neighbors=3)
        self.knn.fit(self.x, self.y)

    def predict(self, image: Image):
        return self.knn.predict(image.reshape(image.shape[0], -1))

    def predict_grid_screenshot(self, screenshot: np.ndarray):
        grid_dim = screenshot.shape[:2]
        screenshot = screenshot.reshape(-1, 110, 110, 3)
        predictions = self.predict(screenshot)
        prediction_grid = np.zeros(grid_dim, dtype=np.int64)
        for i in range(grid_dim[0]):
            for j in range(grid_dim[1]):
                ref = predictions[i * grid_dim[1] + j]
                ref_value = ("empty", "box", "sprite")[ref]
                if self.save_predictions:
                    save_screenshot(screenshot[i * grid_dim[1] + j], os.path.join(self.ref_path, f"../predicted/{ref_value}"), f"{i}_{j}")
                prediction_grid[i, j] = ref
        return prediction_grid

def load_images(ref_path: str):
    """Load the images from the references."""
    images = []
    for i, t in enumerate(os.listdir(ref_path)):
        for f in os.listdir(os.path.join(ref_path, t)):
            if f.endswith(".png"):
                images.append((
                    i, Emulator.read_image(os.path.join(ref_path, t, f))
                ))
    x = np.array([i[1].flatten() for i in images])
    y = np.array([i[0] for i in images])
    return x, y

def main():
    ref_path = "references/sprites/raw"
    x, y = load_images(ref_path)

    # Perform 5 fold cross validation
    knn = KNeighborsClassifier(n_neighbors=3)
    scores = cross_val_score(knn, x, y, cv=5)
    print(scores)

if __name__ == "__main__":
    main()
