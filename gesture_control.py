import mediapipe as mp

mp_hands = mp.solutions.hands

# ---------------- GLOBAL STATE ----------------
prev_gesture = ""
gesture_count = 0
STABLE_FRAMES = 8

prev_landmarks = None
SMOOTHING = 0.6


# ---------------- LANDMARK SMOOTHING ----------------
def smooth_landmarks(hand_landmarks):
    global prev_landmarks

    if prev_landmarks is None:
        prev_landmarks = hand_landmarks
        return hand_landmarks

    for i in range(len(hand_landmarks.landmark)):
        hand_landmarks.landmark[i].x = (
            SMOOTHING * prev_landmarks.landmark[i].x
            + (1 - SMOOTHING) * hand_landmarks.landmark[i].x
        )
        hand_landmarks.landmark[i].y = (
            SMOOTHING * prev_landmarks.landmark[i].y
            + (1 - SMOOTHING) * hand_landmarks.landmark[i].y
        )

    prev_landmarks = hand_landmarks
    return hand_landmarks


# ---------------- FINGER STATE ----------------
def fingers_up(hand_landmarks, hand_label):
    tips = [4, 8, 12, 16, 20]
    fingers = []

    # Thumb
    if hand_label == "Right":
        fingers.append(
            1 if hand_landmarks.landmark[4].x <
            hand_landmarks.landmark[3].x else 0
        )
    else:
        fingers.append(
            1 if hand_landmarks.landmark[4].x >
            hand_landmarks.landmark[3].x else 0
        )

    # Other fingers
    for i in range(1, 5):
        fingers.append(
            1 if hand_landmarks.landmark[tips[i]].y <
            hand_landmarks.landmark[tips[i] - 2].y else 0
        )

    return fingers


# ---------------- HAND ORIENTATION ----------------
def hand_orientation(hand_landmarks):
    wrist_z = hand_landmarks.landmark[
        mp_hands.HandLandmark.WRIST
    ].z

    middle_mcp_z = hand_landmarks.landmark[
        mp_hands.HandLandmark.MIDDLE_FINGER_MCP
    ].z

    # Palm facing camera → wrist is closer
    # Back of hand → middle MCP is closer
    if wrist_z < middle_mcp_z:
        return "PALM"
    else:
        return "BACK"


# ---------------- GESTURE DETECTION ----------------
def detect_gesture(hand_landmarks, label, w, h):
    global prev_gesture, gesture_count

    fingers = fingers_up(hand_landmarks, label)
    pattern = fingers[1:]  # index, middle, ring, pinky

    # -------- PEACE SIGN (NO STABILITY) --------
    if pattern == [1, 1, 0, 0]:
        return "PEACE"

    gesture = "NONE"

    # -------- FIST --------
    if fingers == [0, 0, 0, 0, 0]:
        gesture = "TAKEOFF" if label == "Right" else "LAND"

    # -------- STOP (OPEN PALM) --------
    elif pattern == [1, 1, 1, 1]:
        gesture = "STOP"

    # -------- RIGHT (INDEX) --------
    elif pattern == [1, 0, 0, 0]:
        gesture = "RIGHT"

    # ---------- MIDDLE FINGER EXIT ----------
    elif fingers == [0, 0, 1, 0, 0]:
        orientation = hand_orientation(hand_landmarks)

        if orientation == "BACK":
            return "EXIT"
        else:
            return "NONE"

    # -------- LEFT (PINKY) --------
    elif pattern == [0, 0, 0, 1]:
        gesture = "LEFT"

    # -------- FLIP (INDEX + PINKY) --------
    elif pattern == [1, 0, 0, 1]:
        gesture = "FLIP"

    # -------- STABILITY FILTER --------
    if gesture == prev_gesture:
        gesture_count += 1
    else:
        gesture_count = 0

    prev_gesture = gesture

    if gesture_count > STABLE_FRAMES:
        return gesture

    return "NONE"
