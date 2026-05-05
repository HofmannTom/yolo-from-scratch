import cv2

# ----------------------------
# Draw_boxes Funktion
# ----------------------------
def draw_boxes(img, boxes, conf_thresh=0.5):
    h, w = img.shape

    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    for box in boxes:
        if len(box) == 6:
            class_id, x_c, y_c, bw, bh, conf = box
        else:
            class_id, x_c, y_c, bw, bh = box
            conf = 1.0

        if conf < conf_thresh:
            continue

        if bw <= 0 or bh <= 0:
            continue

        x1 = int((x_c - bw / 2) * w)
        y1 = int((y_c - bh / 2) * h)
        x2 = int((x_c + bw / 2) * w)
        y2 = int((y_c + bh / 2) * h)

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return img

# ----------------------------
# IoU Funktion
# ----------------------------
def iou(box1, box2):
    # box = (x1, y1, x2, y2)

    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    # intersection area
    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    inter_area = inter_w * inter_h

    # areas
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union_area = box1_area + box2_area - inter_area

    if union_area == 0:
        return 0

    return inter_area / union_area


# ----------------------------
# Loss-Funktion
# ----------------------------
def yolo_loss(pred, target):
    obj_mask = target[..., 4] == 1
    noobj_mask = target[..., 4] == 0

    # Box loss
    box_loss = ((pred[..., :4] - target[..., :4])**2)
    box_loss = box_loss[obj_mask].sum()

    # Confidence loss (obj)
    conf_loss_obj = ((pred[..., 4] - target[..., 4])**2)
    conf_loss_obj = conf_loss_obj[obj_mask].sum()

    # Confidence loss (no obj)
    conf_loss_noobj = ((pred[..., 4] - target[..., 4])**2)
    conf_loss_noobj = conf_loss_noobj[noobj_mask].sum()

    loss = (
        5 * box_loss +
        conf_loss_obj +
        0.5 * conf_loss_noobj
    )

    return loss