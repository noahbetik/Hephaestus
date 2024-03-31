class GestureProcessing:
    def __init__(self, gesture_model):
        # Load the gesture recognition model with the given settings
        self.gesture_model = gesture_model
        pass

    # Detects if it's a single or dual hand gesture
    def detect_single_or_dual(self, left_hand_gesture_id, right_hand_gesture_id):
        # Define recognized dual gesture combinations
        dual_gesture_combinations = {
            (1, 6): "One finger and a fist",
            (2, 6): "Two fingers and a fist",
            (4, 6): "Three fingers and a fist",
            (10, 11): "Illuminati",
            (8, 7): "L Shape",
        }

        # Check if both hands are involved in a gesture
        if left_hand_gesture_id is not None and right_hand_gesture_id is not None:
            # Try to get the dual gesture combination
            dual_gesture = dual_gesture_combinations.get(
                (left_hand_gesture_id, right_hand_gesture_id)
            )

            # If a recognized dual gesture is detected
            if dual_gesture:
                return dual_gesture
            else:
                # If a dual gesture is detected but not recognized
                print("Invalid gesture pair")
                return "Invalid gesture pair"
        else:
            # If it's not a dual gesture
            return False

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
