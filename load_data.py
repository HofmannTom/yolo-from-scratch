import os
import cv2
import matplotlib.pyplot as plt

# Pfad zum Dataset 
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "data", "Dental OPG (Classification)/BDC-BDR")

def load_images_from_folder(folder):
    images = []
    labels = []
    print(os.listdir(folder))
    
    for label_name in os.listdir(folder):
        label_path = os.path.join(folder, label_name)
        
        print(os.path.isdir(label_path))
        if not os.path.isdir(label_path):
            continue
        
        for file in os.listdir(label_path):
            img_path = os.path.join(label_path, file)
            
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            
            if img is not None:
                images.append(img)
                labels.append(label_name)
    
    return images, labels


def show_samples(images, labels, num_samples=5):
    num_samples = min(num_samples, len(images))

    for i in range(num_samples):
        plt.imshow(images[i], cmap='gray')
        plt.title(f"Label: {labels[i]}")
        plt.axis('off')
        plt.show()


if __name__ == "__main__":
    images, labels = load_images_from_folder(DATASET_PATH)
    print(DATASET_PATH)
    print(os.path.exists(DATASET_PATH))
    
    print(f"Loaded {len(images)} images")
    
    show_samples(images, labels)