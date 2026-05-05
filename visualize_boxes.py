import os
import cv2
import matplotlib.pyplot as plt

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


def draw_boxes(img, boxes):
    h, w = img.shape[:2]

    for box in boxes:
        class_id, x_c, y_c, bw, bh = box

        # YOLO -> Pixel
        x1 = int((x_c - bw / 2) * w)
        y1 = int((y_c - bh / 2) * h)
        x2 = int((x_c + bw / 2) * w)
        y2 = int((y_c + bh / 2) * h)

        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)

    return img


def load_sample():
    for file in os.listdir(IMAGES_PATH):
        if file.endswith(".jpg"):
            img_path = os.path.join(IMAGES_PATH, file)
            label_path = os.path.join(LABELS_PATH, file.replace(".jpg", ".txt"))

            gray_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img_color = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)

            boxes = []

            if os.path.exists(label_path):
                with open(label_path, "r") as f:
                    for line in f.readlines():
                        boxes.append(list(map(float, line.strip().split())))

            return img_color, boxes


if __name__ == "__main__":
    img, boxes = load_sample()

    img_with_boxes = draw_boxes(img, boxes)

    plt.imshow(cv2.cvtColor(img_with_boxes, cv2.COLOR_BGR2RGB))
    plt.title("Dental X-ray with Bounding Boxes")
    plt.axis("off")
    plt.show()