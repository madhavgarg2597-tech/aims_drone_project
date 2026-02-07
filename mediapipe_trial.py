import cv2
import mediapipe as mp
import math

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh

face_detection = mp_face.FaceDetection(min_detection_confidence=0.5)
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

# -------- GLOBAL STATE --------
eye_mode = False
peace_frames = 0
peace_cooldown = 0
prev_gesture = ""
gesture_count = 0
STABLE_FRAMES = 8
joystick_mode = False
joy_frames = 0

alpha = 0.2
h_smooth = 0.5
v_smooth = 0.5

# -------- FUNCTIONS --------
def get_finger_angle(hand_landmarks, w, h):
    tip = hand_landmarks.landmark[8]  # Index Tip
    base = hand_landmarks.landmark[5] # Index MCP (Knuckle)
    x1, y1 = int(base.x * w), int(base.y * h)
    x2, y2 = int(tip.x * w), int(tip.y * h)
    angle = math.degrees(math.atan2(y1 - y2, x2 - x1))
    return angle, (x1, y1), (x2, y2)

def draw_joystick(img, direction, w):
    cx, cy = w - 100, 100
    r_outer, r_inner = 50, 20
    cv2.circle(img, (cx, cy), r_outer, (200, 200, 200), 2)
    kx, ky = cx, cy
    offset = 30
    color = (255, 0, 255)
    
    if direction == "UP": ky -= offset
    elif direction == "DOWN": ky += offset
    elif direction == "LEFT": kx -= offset
    elif direction == "RIGHT": kx += offset
    else: color = (100, 100, 100)
    
    cv2.circle(img, (kx, ky), r_inner, color, -1)
    cv2.putText(img, "JOYSTICK", (cx-40, cy+75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

def fingers_up(hand_landmarks, hand_label):
    tips = [4, 8, 12, 16, 20]
    fingers = []
    if hand_label == "Right":
        fingers.append(1 if hand_landmarks.landmark[4].x < hand_landmarks.landmark[3].x else 0)
    else:
        fingers.append(1 if hand_landmarks.landmark[4].x > hand_landmarks.landmark[3].x else 0)
    for i in range(1, 5):
        fingers.append(1 if hand_landmarks.landmark[tips[i]].y < hand_landmarks.landmark[tips[i] - 2].y else 0)
    return fingers

def hand_orientation(hand_landmarks):
    wrist_z = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].z
    middle_mcp_z = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].z
    # PALM = palm facing camera; BACK = knuckles facing camera
    return "PALM" if middle_mcp_z < wrist_z else "BACK"

cap = cv2.VideoCapture(0)

with mp_hands.Hands(model_complexity=0, min_detection_confidence=0.7, min_tracking_confidence=0.7) as hands:
    while cap.isOpened():
        success, image = cap.read()
        if not success: continue

        image = cv2.flip(image, 1)
        h, w, _ = image.shape
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        results = hands.process(image_rgb)
        face_results = face_detection.process(image_rgb)
        gesture = "NONE"

        if results.multi_hand_landmarks and face_results.detections:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                label = handedness.classification[0].label
                orientation = hand_orientation(hand_landmarks)
                
                # Face Box calculation
                bbox = face_results.detections[0].location_data.relative_bounding_box
                fx, fy, fw, fh = int(bbox.xmin * w), int(bbox.ymin * h), int(bbox.width * w), int(bbox.height * h)
                box_x1, box_x2 = (fx + fw + 20, fx + fw + 220) if label == "Right" else (fx - 220, fx - 20)
                box_y1, box_y2 = fy, fy + 200
                
                hx, hy = int(hand_landmarks.landmark[0].x * w), int(hand_landmarks.landmark[0].y * h)
                inside_box = box_x1 < hx < box_x2 and box_y1 < hy < box_y2
                
                cv2.rectangle(image, (box_x1, box_y1), (box_x2, box_y2), (0, 255, 0) if inside_box else (0, 0, 255), 3)
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                fingers = fingers_up(hand_landmarks, label)
                pattern = fingers[1:] # Index, Middle, Ring, Pinky

                # TOGGLE Logic
                if pattern == [1, 1, 0, 0] and peace_cooldown == 0:
                    joy_frames += 1
                else: joy_frames = 0
                if joy_frames > STABLE_FRAMES:
                    joystick_mode = not joystick_mode
                    joy_frames, peace_cooldown = 0, 40

                if inside_box:
                    if joystick_mode:
                        # Index up [1], Others down [0,0,0] + Knuckles facing camera
                        if pattern == [1, 0, 0, 0] and orientation == "BACK":
                            angle, p1, p2 = get_finger_angle(hand_landmarks, w, h)
                            cv2.arrowedLine(image, p1, p2, (255, 0, 255), 4)
                            
                            if -45 < angle < 45: gesture = "RIGHT"
                            elif 45 <= angle < 135: gesture = "UP"
                            elif -135 < angle <= -45: gesture = "DOWN"
                            else: gesture = "LEFT"
                    else:
                        # Normal Mode
                        if fingers == [0,0,0,0,0]:
                            gesture = "LAND" if orientation == "PALM" else "TAKEOFF"
                        elif pattern == [1,1,1,1]: gesture = "STOP"
                        elif pattern == [1,0,0,0]: gesture = "RIGHT"
                        elif pattern == [1,0,0,1]: gesture = "FLIP"

                if gesture == prev_gesture and gesture != "NONE":
                    gesture_count += 1
                else: gesture_count = 0
                prev_gesture = gesture
                
                if gesture_count > STABLE_FRAMES:
                    cv2.putText(image, gesture, (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3)

        if joystick_mode:
            cv2.putText(image, "JOYSTICK MODE: KNUCKLE SIDE", (w//2-200, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
            draw_joystick(image, gesture if gesture != "NONE" else "IDLE", w)

        cv2.imshow("Drone Control HUD", image)
        if cv2.waitKey(5) & 0xFF == 27: break

cap.release()
cv2.destroyAllWindows()