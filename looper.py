"""Loop controller to record a chord sequence and play it back."""
import time


class Loop():
    """Record chords with timestamps and replay them in a loop. Each
    detected fist changes the state from idle, to recording, to playing,
    and back to idle."""
    def __init__(self):
        """Start in the idle state."""
        self.state = "idle"
        self.start_time = None
        self.end_time = None
        self.duration = None
        self.key_history = []
        self.last_note = None


    def on_fist(self):
        """State transitions when fist detected.
        
        Return:
            str: new state
        """
        if self.state == "idle":
            self.state = "recording"
            self.start_time = time.monotonic()
            self.key_history = []
            self.last_note = None
        elif self.state == "recording":
            self.state = "playing"
            self.end_time = time.monotonic()
            self.duration = self.end_time - self.start_time
        else:
            self.state = "idle"

        return self.state


    def save_chords(self, notes):
        """Record new chords.
        
        Arg:
            notes (tuple): MIDI note numbers currently being played.
            
        Return:
            list: full history (notes, offset) recorded so far."""
        if notes != self.last_note:
            self.key_history.append((notes, time.monotonic() - self.start_time))
            self.last_note = notes
        return self.key_history

    def playback(self):
        """Find the chord that should be played right now in the loop.
        
        Return:
            tuple: MIDI note numbers to play (empty for silence)
        """
        # wrap elapsed time around loop length
        position = (time.monotonic() - self.end_time) % self.duration
        current = ()
        for notes, offset in self.key_history:
            if offset <= position:
                current = notes
        return current