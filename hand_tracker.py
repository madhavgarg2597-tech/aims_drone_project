import mediapipe as mp
from gesture_control import smooth_landmarks

mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_detection

hands = mp_hands.Hands(
    max_num_hands=1,
    model_complexity=0,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

face_detection = mp_face.FaceDetection(min_detection_confidence=0.7)

def detect(frame):
    rgb = frame[:, :, ::-1]
    return hands.process(rgb), face_detection.process(rgb)

def preprocess(hand_landmarks):
    return smooth_landmarks(hand_landmarks)
