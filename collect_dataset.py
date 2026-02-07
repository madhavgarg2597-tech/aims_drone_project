import cv2
import os
import time

# ---------------- CONFIG ----------------
DATASET_DIR = "dataset"
IMG_SIZE = 200
SAVE_DELAY = 0.15  # seconds between saves

# Classes mapping
classes = {
    "f": "fist",
    "p": "palm",
    "e": "peace",
    "i": "index",
    "k": "pinky"
}

# Create folders if not exist
for cls in classes.values():
    os.makedirs(os.path.join(DATASET_DIR, cls), exist_ok=True)

# ---------------- CAMERA ----------------
cap = cv2.VideoCapture(2)  # change index if using phone cam

last_save = time.time()

print("""
CONTROLS:
f = FIST
p = PALM
e = PEACE
i = INDEX
k = PINKY
q = QUIT
""")


while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    # -------- ROI BOX (RIGHT SIDE OF FACE AREA) --------
    box_size = 260
    x1 = w//2 - box_size//2
    y1 = h//2 - box_size//2
    x2 = x1 + box_size
    y2 = y1 + box_size

    roi = frame[y1:y2, x1:x2]

    # Convert to grayscale
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))

    # Draw box
    cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)

    cv2.putText(
        frame,
        "Place HAND inside box",
        (x1, y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255,255,255),
        2
    )

    cv2.imshow("Dataset Collector", frame)
    cv2.imshow("ROI", gray)

    key = cv2.waitKey(1) & 0xFF

    # -------- SAVE IMAGE --------
    if key in [ord(k) for k in classes.keys()]:
        if time.time() - last_save > SAVE_DELAY:
            label = classes[chr(key)]
            folder = os.path.join(DATASET_DIR, label)
            count = len(os.listdir(folder))

            filename = f"{label}_{count}.jpg"
            path = os.path.join(folder, filename)

            cv2.imwrite(path, gray)
            print(f"Saved: {path}")
            last_save = time.time()

    elif key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
