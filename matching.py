import torch
from src.utils import iou


# -------------------------
# YOLO-LIGHT MATCHING
# -------------------------
def force_match(pred_boxes, gt_boxes, image_size=256):
    """
    pred_boxes: (10, 5)
    gt_boxes:   (N, 5)
    """

    matched_preds = []
    matched_gts = []

    for gt in gt_boxes:

        gt_xyxy = convert_to_xyxy(gt, image_size)

        best_iou = -1
        best_pred = None

        for pred in pred_boxes:

            pred_xyxy = convert_to_xyxy(pred, image_size)

            score = iou(pred_xyxy, gt_xyxy)

            if score > best_iou:
                best_iou = score
                best_pred = pred

        matched_preds.append(best_pred)
        matched_gts.append(gt)

    return torch.stack(matched_preds), torch.stack(matched_gts)

# -------------------------
# Match_prediction
# -------------------------
def match_predictions(pred_boxes, gt_boxes, iou_threshold=0.0):
    """
    pred_boxes: (10, 5)
    gt_boxes:   (N, 5)
    """

    matched_pred = []
    matched_gt = []

    used_preds = set()

    for gt in gt_boxes:
        best_iou = 0
        best_idx = -1

        # GT Box in Pixel umrechnen (falls nötig)
        gt_box = convert_to_xyxy(gt)
        
        for i, pred in enumerate(pred_boxes):
            if i in used_preds:
                continue

            pred_box = convert_to_xyxy(pred)
            print("GT raw:", gt)
            print("GT xyxy:", convert_to_xyxy(gt))

            print("Pred raw:", pred)
            print("Pred xyxy:", convert_to_xyxy(pred))

            score = iou(pred_box, gt_box)

            if score > best_iou:
                best_iou = score
                best_idx = i

        # wenn gute Übereinstimmung gefunden
        if best_iou > iou_threshold:
            matched_pred.append(best_idx)
            matched_gt.append(gt)
            used_preds.add(best_idx)

    return matched_pred, matched_gt

# ----------------------------
# Konvertier funktion für IoU Funktion
# ----------------------------
def convert_to_xyxy(box, img_size=256):
    _, x, y, w, h = box

    x1 = (x - w / 2) * img_size
    y1 = (y - h / 2) * img_size
    x2 = (x + w / 2) * img_size
    y2 = (y + h / 2) * img_size

    return [x1, y1, x2, y2]