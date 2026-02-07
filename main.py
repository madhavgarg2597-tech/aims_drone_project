import cv2
import mediapipe as mp
import math
import time

from gesture_control import detect_gesture, smooth_landmarks
from utils import draw_text, draw_control_box

# -------------------- MEDIAPIPE --------------------
mp_face = mp.solutions.face_detection
face_detection = mp_face.FaceDetection(min_detection_confidence=0.7)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    model_complexity=0,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# -------------------- STATES --------------------
joystick_mode = False
peace_cooldown = 0

box_locked = False
locked_box = None
joy_center = None

last_hx = None
last_hy = None

DEAD_ZONE = 20
MAX_RADIUS = 120

gesture_start_time = None
last_gesture = None
GESTURE_HOLD_TIME = 4  # seconds

# -------------------- CAMERA --------------------
cap = cv2.VideoCapture(0)

while True:
    ret, image = cap.read()
    if not ret:
        break

    image = cv2.flip(image, 1)
    h, w, _ = image.shape

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    face_results = face_detection.process(rgb)

    if peace_cooldown > 0:
        peace_cooldown -= 1

    # -------------------- HAND DETECTION --------------------
    if results.multi_hand_landmarks and face_results.detections:
        hand_landmarks = results.multi_hand_landmarks[0]
        handedness = results.multi_handedness[0]
        label = handedness.classification[0].label

        hand_landmarks = smooth_landmarks(hand_landmarks)

        # ---------- FACE-BASED BOX (ONLY WHEN NOT LOCKED) ----------
        if not joystick_mode:
            detection = face_results.detections[0]
            bbox = detection.location_data.relative_bounding_box

            fx = int(bbox.xmin * w)
            fy = int(bbox.ymin * h)
            fw = int(bbox.width * w)
            fh = int(bbox.height * h)

            if label == "Right":
                box_x1, box_x2 = fx + fw + 20, fx + fw + 220
            else:
                box_x1, box_x2 = fx - 220, fx - 20

            face_center_y = fy + fh // 2
            box_y1 = face_center_y - 100
            box_y2 = face_center_y + 100

            current_box = (box_x1, box_y1, box_x2, box_y2)
        else:
            current_box = locked_box
            box_x1, box_y1, box_x2, box_y2 = locked_box

        # ---------- INDEX FINGER ----------
        tip = hand_landmarks.landmark[8]
        hx, hy = int(tip.x * w), int(tip.y * h)
        last_hx, last_hy = hx, hy

        inside_box = (
            box_x1 < hx < box_x2 and
            box_y1 < hy < box_y2
        )

        draw_control_box(image, current_box, inside_box)

        mp.solutions.drawing_utils.draw_landmarks(
            image,
            hand_landmarks,
            mp.solutions.hands.HAND_CONNECTIONS
        )

        cv2.putText(
            image,
            label,
            (box_x1, box_y2 + 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        # ---------- GESTURE ----------
        if inside_box or joystick_mode:
            gesture = detect_gesture(hand_landmarks, label, w, h)

            if gesture == "EXIT":
                bye_frame = image.copy()
                cv2.putText(
                    bye_frame,
                    "BYE!!",
                    (w // 2 - 160, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    4,
                    (0, 0, 255),
                    10
                )
                cv2.imshow("VisionPilot", bye_frame)
                cv2.waitKey(1000)

                cap.release()
                cv2.destroyAllWindows()
                exit()

            # ---- PEACE TOGGLE ----
            if gesture == "PEACE" and peace_cooldown == 0:
                joystick_mode = not joystick_mode
                peace_cooldown = 25

                if joystick_mode:
                    locked_box = current_box
                    joy_center = (
                        (box_x1 + box_x2) // 2,
                        (box_y1 + box_y2) // 2
                    )
                    print("JOYSTICK MODE ON")
                else:
                    locked_box = None
                    joy_center = None
                    print("JOYSTICK MODE OFF")

            # ---------- NORMAL MODE ----------
            elif not joystick_mode and gesture != "NONE":
                print("COMMAND:", gesture)
                draw_text(image, gesture, 70)

    # -------------------- JOYSTICK MODE --------------------
    if joystick_mode and locked_box is not None and last_hx is not None:
        box_x1, box_y1, box_x2, box_y2 = locked_box
        draw_control_box(image, locked_box, True)

        dx = last_hx - joy_center[0]
        dy = last_hy - joy_center[1]

        dist = math.sqrt(dx * dx + dy * dy)
        dist = min(dist, MAX_RADIUS)

        speed = 0 if dist < DEAD_ZONE else int((dist / MAX_RADIUS) * 100)

        if abs(dx) > abs(dy):
            cmd = "STRAFE_RIGHT" if dx > 0 else "STRAFE_LEFT"
        else:
            cmd = "FORWARD" if dy < 0 else "BACKWARD"

        print(f"COMMAND: {cmd} | SPEED: {speed}%")

        # ---------- HUD ----------
        hud_center = (w - 120, 120)
        hud_radius = 50

        cv2.circle(image, hud_center, hud_radius, (140, 140, 140), 2)
        cv2.circle(image, hud_center, int(DEAD_ZONE * 0.5), (180, 180, 180), 1)

        if speed > 0:
            norm = max(math.sqrt(dx * dx + dy * dy), 1)
            ux, uy = dx / norm, dy / norm
            arrow_len = int((speed / 100) * hud_radius)

            end_x = int(hud_center[0] + ux * arrow_len)
            end_y = int(hud_center[1] + uy * arrow_len)

            cv2.arrowedLine(
                image,
                hud_center,
                (end_x, end_y),
                (0, 255, 0),
                3,
                tipLength=0.3
            )

        cv2.putText(
            image,
            f"Speed : {speed}%",
            (hud_center[0] - 60, hud_center[1] + 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

        draw_text(image, "JOYSTICK MODE", 35)

    cv2.imshow("VisionPilot", image)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
