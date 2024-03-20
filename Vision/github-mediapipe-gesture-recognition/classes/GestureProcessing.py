class GestureProcessing:
    def __init__(self, gesture_model):
        # Load the gesture recognition model with the given settings
        self.gesture_model = gesture_model
        pass

    # Detects if it's a single or dual hand gesture
    def detect_single_or_dual(self):
        pass

    ## STAGES #############
    # Locks in the gesture
    def lock_in_stage(self):
        pass

    # End gesture
    def active_stage(self):
        pass

    # End gesture
    def end_stage(self):
        pass

    ## COMMAND SENDING #############
    # Sends start command
    def sends_start_command(self):
        pass

    # Sends active commands
    def send_active_commands(self):
        pass

    # Sends end command
    def send_end_command(self):
        pass

    # Sends One-hit Command
    def send_one_hit_command(self):
        pass
