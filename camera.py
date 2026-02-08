import cv2

def init_camera(index=0):
    return cv2.VideoCapture(index)

def read_frame(cap):
    ret, frame = cap.read()
    if not ret:
        return None
    return cv2.flip(frame, 1)
