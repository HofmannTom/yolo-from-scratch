import os
import cv2

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

DATASET_PATH = os.path.join(
    BASE_DIR,
    "data",
    "Dental OPG (Object Detection)",
    "Augmented Dataset",
    "train"
)

IMAGES_PATH = os.path.join(DATASET_PATH, "images")
LABELS_PATH = os.path.join(DATASET_PATH, "labels")


def load_data():
    data = []

    for file in os.listdir(IMAGES_PATH):
        if file.endswith(".jpg"):
            img_path = os.path.join(IMAGES_PATH, file)
            label_path = os.path.join(LABELS_PATH, file.replace(".jpg", ".txt"))

            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            boxes = []
            if os.path.exists(label_path):
                with open(label_path, "r") as f:
                    for line in f.readlines():
                        values = list(map(float, line.strip().split()))
                        boxes.append(values)

            data.append((img, boxes))

    return data


if __name__ == "__main__":
    dataset = load_data()
    print(f"Loaded {len(dataset)} samples")

    # Debug: erstes Sample anschauen
    img, boxes = dataset[0]
    print("Boxes:", boxes)