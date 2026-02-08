import cv2
from camera import init_camera, read_frame
from hand_tracker import detect, preprocess
from box_manager import compute_control_box, inside_box
from joystick import compute_command
from hud import draw_joystick_hud
from state import AppState
from gesture_control import detect_gesture
from utils import draw_text, draw_control_box

cap = init_camera()
state = AppState()

while True:
    frame = read_frame(cap)
    if frame is None:
        break

    h, w, _ = frame.shape
    hand_res, face_res = detect(frame)

    if state.peace_cooldown > 0:
        state.peace_cooldown -= 1

    if hand_res.multi_hand_landmarks and face_res.detections:
        hand = preprocess(hand_res.multi_hand_landmarks[0])
        label = hand_res.multi_handedness[0].classification[0].label

        if not state.joystick_mode:
            bbox = face_res.detections[0].location_data.relative_bounding_box
            face_box = (
                int(bbox.xmin * w),
                int(bbox.ymin * h),
                int(bbox.width * w),
                int(bbox.height * h),
            )
            state.locked_box = compute_control_box(face_box, label, w, h)

        tip = hand.landmark[8]
        hx, hy = int(tip.x * w), int(tip.y * h)
        state.last_hx, state.last_hy = hx, hy

        active = inside_box(hx, hy, state.locked_box)
        draw_control_box(frame, state.locked_box, active)

        gesture = detect_gesture(hand, label, w, h)

        if gesture == "PEACE" and state.peace_cooldown == 0:
            state.joystick_mode = not state.joystick_mode
            state.peace_cooldown = 25
            if state.joystick_mode:
                x1,y1,x2,y2 = state.locked_box
                state.joy_center = ((x1+x2)//2, (y1+y2)//2)
            else:
                state.joy_center = None

        elif not state.joystick_mode and active and gesture != "NONE":
            draw_text(frame, gesture, 70)

    if state.joystick_mode and state.joy_center:
        cmd, speed, dx, dy = compute_command(
            state.last_hx, state.last_hy, state.joy_center
        )
        print(f"COMMAND: {cmd} | SPEED: {speed}%")
        draw_joystick_hud(frame, dx, dy, speed, w)
        draw_text(frame, "JOYSTICK MODE", 35)

    cv2.imshow("VisionPilot", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
