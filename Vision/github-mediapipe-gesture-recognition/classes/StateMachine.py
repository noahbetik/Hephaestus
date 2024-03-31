class StateMachine:
    def __init__(self):
        # 0 = nothing, 1 = lock-in, 2 = active, 3 = end
        self.state = 0
        self.lock_in_frames_needed = 20  # Number of frames to confirm a gesture
        self.lock_in_confidence = 0.60
        self.current_lock_in_frames = 0  # Current frame count for lock-in

    def update(self, gesture_id, confidence, is_dual):
        # Transition logic
        if self.state == 0:  # Nothing
            if gesture_id is not None and confidence > self.lock_in_confidence:
                self.current_lock_in_frames += 1
                if self.current_lock_in_frames >= self.lock_in_frames_needed:
                    self.state = 1  # Move to lock-in state
                    self.current_lock_in_frames = 0  # Reset frame counter for lock-in
        elif self.state == 1:  # Lock-in
            # Assuming we have logic to confirm lock-in; transition to active
            self.state = 2
        elif self.state == 2:  # Active
            if gesture_id is None:  # User ends gesture
                self.state = 3  # Move to end state
        elif self.state == 3:  # End
            # Perform end actions
            self.state = 0  # Return to nothing state
