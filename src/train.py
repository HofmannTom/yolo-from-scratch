import os
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torch.utils.data import DataLoader
from src.dataset import DentalDataset
from src.model import SimpleYOLO
from src.utils import draw_boxes
from src.utils import yolo_loss
from src.utils import nms


# ----------------------------
# Paths
# ----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(__file__))

IMAGES_PATH = os.path.join(BASE_DIR, "data", "Dental OPG (Object Detection)", "Augmented Dataset", "train", "images")
LABELS_PATH = os.path.join(BASE_DIR, "data", "Dental OPG (Object Detection)", "Augmented Dataset", "train", "labels")

# ----------------------------
# Collate Function
# ----------------------------
def collate_fn(batch):
    imgs = []
    targets = []

    for img, boxes in batch:
        imgs.append(img)
        targets.append(boxes)

    imgs = torch.stack(imgs)          # (B, 1, 256, 256)
    targets = torch.stack(targets)    # (B, 10, 5)

    return imgs, targets

# ----------------------------
# Setup
# ----------------------------
dataset = DentalDataset(IMAGES_PATH, LABELS_PATH)

loader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True,
    collate_fn=collate_fn
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SimpleYOLO().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# ----------------------------
# Training Loop
# ----------------------------
EPOCHS = 20

for epoch in range(EPOCHS):
    total_loss = 0
    model.train()

    for imgs, targets in loader:
        imgs = imgs.to(device)
        targets = targets.to(device)

        # Forward
        preds = model(imgs)  # (B, 7, 7, 5)

        # Loss (batch-wise sauber)
        loss = 0
        for i in range(preds.shape[0]):
            loss += yolo_loss(preds[i], targets[i])

        loss = loss / preds.shape[0]

        # Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch} Loss: {total_loss:.4f}")

torch.save(model.state_dict(), "model.pt") #Model_speichern

# ----------------------------
# Visualisierung (FINAL)
# ----------------------------
print("Visualisiere Predictions...")

model.eval()

with torch.no_grad():
    for imgs, targets in loader:

        imgs = imgs.to(device)
        preds = model(imgs)

        # Erstes Bild im Batch
        img = imgs[0].cpu().squeeze().numpy()
        pred_grid = preds[0].cpu()
        target_grid = targets[0]

        GRID_SIZE = 7

        pred_boxes = []
        target_boxes = []

        # ----------------------------
        # Grid → Boxen umwandeln
        # ----------------------------
        for gy in range(GRID_SIZE):
            for gx in range(GRID_SIZE):

                # -------- PRED --------
                px, py, pw, ph, pconf = pred_grid[gy, gx]

                if pconf > 0.5:
                    x = (gx + px.item()) / GRID_SIZE
                    y = (gy + py.item()) / GRID_SIZE
                    w = pw.item()
                    h = ph.item()

                    pred_boxes.append([0, x, y, w, h, pconf.item()])

                # -------- GT --------
                tx, ty, tw, th, tconf = target_grid[gy, gx]

                if tconf > 0:
                    x = (gx + tx.item()) / GRID_SIZE
                    y = (gy + ty.item()) / GRID_SIZE
                    w = tw.item()
                    h = th.item()

                    target_boxes.append([0, x, y, w, h])

        # Tensor bauen
        pred_boxes = torch.tensor(pred_boxes) if len(pred_boxes) > 0 else torch.zeros((0,6))
        target_boxes = torch.tensor(target_boxes) if len(target_boxes) > 0 else torch.zeros((0,5))

        # ----------------------------
        # NMS anwenden
        # ----------------------------
        print("Predictions before NMS:", len(pred_boxes))

        if len(pred_boxes) > 0:
            keep = nms(pred_boxes, iou_threshold=0.5)
            pred_boxes = pred_boxes[keep]

        print("Predictions after NMS:", len(pred_boxes))

        # ----------------------------
        # Zeichnen
        # ----------------------------
        img_pred = draw_boxes(img.copy(), pred_boxes.numpy())
        img_gt = draw_boxes(img.copy(), target_boxes.numpy())

        # ----------------------------
        # Plot + Save
        # ----------------------------
        plt.figure(figsize=(10,5))

        plt.subplot(1,2,1)
        plt.title("Predictions (NMS)")
        plt.imshow(img_pred)
        plt.axis("off")

        plt.subplot(1,2,2)
        plt.title("Ground Truth")
        plt.imshow(img_gt)
        plt.axis("off")

        save_path = "results/prediction.png"
        plt.savefig(save_path)
        print(f"Bild gespeichert unter: {save_path}")

        plt.show()

        break