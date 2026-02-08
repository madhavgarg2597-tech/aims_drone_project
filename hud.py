import cv2
import math

def draw_joystick_hud(frame, dx, dy, speed, w):
    center = (w - 120, 120)
    radius = 50

    cv2.circle(frame, center, radius, (140,140,140), 2)
    cv2.circle(frame, center, 10, (180,180,180), 1)

    if speed > 0:
        norm = max(math.sqrt(dx*dx + dy*dy), 1)
        ux, uy = dx / norm, dy / norm
        length = int((speed / 100) * radius)
        end = (int(center[0] + ux*length), int(center[1] + uy*length))
        cv2.arrowedLine(frame, center, end, (0,255,0), 3)

    cv2.putText(
        frame,
        f"Speed : {speed}%",
        (center[0]-60, center[1]+80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0,255,0),
        2
    )