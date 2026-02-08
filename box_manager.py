def compute_control_box(face_bbox, hand_label, w, h):
    fx, fy, fw, fh = face_bbox
    if hand_label == "Right":
        x1, x2 = fx + fw + 20, fx + fw + 220
    else:
        x1, x2 = fx - 220, fx - 20
    cy = fy + fh // 2
    return (x1, cy - 100, x2, cy + 100)

def inside_box(x, y, box):
    x1, y1, x2, y2 = box
    return x1 < x < x2 and y1 < y < y2
