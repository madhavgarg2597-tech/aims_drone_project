import cv2

def draw_text(image, text, y):
    cv2.putText(image, text, (30, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)


def draw_control_box(image, box, inside):
    x1, y1, x2, y2 = box
    color = (0,255,0) if inside else (0,0,255)
    cv2.rectangle(image, (x1,y1), (x2,y2), color, 3)
