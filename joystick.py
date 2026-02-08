import math

DEAD_ZONE = 20
MAX_RADIUS = 120

def compute_command(hx, hy, center):
    dx = hx - center[0]
    dy = hy - center[1]

    dist = min(math.sqrt(dx*dx + dy*dy), MAX_RADIUS)
    speed = 0 if dist < DEAD_ZONE else int((dist / MAX_RADIUS) * 100)

    if abs(dx) > abs(dy):
        cmd = "STRAFE_RIGHT" if dx > 0 else "STRAFE_LEFT"
    else:
        cmd = "FORWARD" if dy < 0 else "BACKWARD"

    return cmd, speed, dx, dy