class AppState:
    def __init__(self):
        self.joystick_mode = False
        self.peace_cooldown = 0
        self.locked_box = None
        self.joy_center = None
        self.last_hx = None
        self.last_hy = None
