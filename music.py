import fluidsynth


class MusicElement:
    """one sound to play"""

    def __init__(self):
        self.fs = fluidsynth.Synth()
        self.fs.start(driver="pulseaudio")   # if this fails, try "alsa"
        sfid = self.fs.sfload("/usr/share/sounds/sf2/FluidR3_GM.sf2")
        self.fs.program_select(0, sfid, 0, 48)
        # fs.cc(channel, controller , value)
        self.fs.cc(0, 91, 0) # reverb controller
        self.fs.cc(0, 93, 0) # chorus controller
        self.fs.cc(0, 72, 0) # release time

        self.root = 60; # root note, middle c as default for now
        self.current_note = None # interval note
        self.velocity = 100

        self.mode = "major" # default to major
        # maps intervals to semitones above any starting note
        self.gesture_to_semitones = {
            "major": {
                "interval_2": 2, # major second
                "interval_3": 4, # major third
                "interval_4": 5, # perfect fourth
                "interval_5": 7, # perfect fifth
                "interval_6": 9, # major sixth
                "interval_7": 11, # major seventh
                "interval_8": 12 # perfect octave
            },
            "minor": {
                "interval_2": 1, # minor second
                "interval_3": 3, # minor third
                "interval_5": 6, # minor fifth, same as diminished fifth
                "interval_7": 10 # minor seventh
            }
        }

    def play(self, gesture):
        # play the sound
        if gesture == "major" or gesture == "minor":
            self.mode = gesture
            return

        offset = self.gesture_to_semitones.get(self.mode, {}).get(gesture)

        if offset is None: 
            self.stop()
        else: 
            note = self.root + offset
            if note == self.current_note: 
                return
            self.stop()
            self.fs.noteon(0, self.root, self.velocity)
            self.fs.noteon(0, note, self.velocity)
            self.current_note = note

    def stop(self):
        if self.current_note is not None: 
            self.fs.noteoff(0, self.root)
            self.fs.noteoff(0,self.current_note)
            self.fs.cc(0,120,0)
        self.current_note = None
