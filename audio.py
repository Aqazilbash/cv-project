"""Setting up Fluidsynth and outputting audio based on notes given."""

import fluidsynth



def make_synth(sf2="/usr/share/sounds/sf2/FluidR3_GM.sf2", 
               driver="pulseaudio",
               program=48):
    """Initialize the synth.
    
    Args:
        sf2 (str): path to SoundFont file  
        driver (str): audio driver dependent on OS
        program (int): MIDI instrument number
    """
    fs = fluidsynth.Synth()
    fs.start(driver=driver)   # if this fails, try "alsa"
    sfid = fs.sfload(sf2)
    for channel in (0,1): # separate audio channels for live and playback
        fs.program_select(channel, sfid, 0, program)
    return fs

class ChordPlayer:
    """Plays one chord at a time."""
    def __init__(self, synth, channel=0, velocity=90):
        """Initializes module and Fluidsynth parameters."""
        self.fs = synth
        self.channel = channel
        self.velocity = velocity
        self.current = ()          # tuple of MIDI notes currently sounding

    def set_chord(self, notes):
        """Switch chord being played to the given, new notes. If the
        chord is the same, keep playing the current one.
        
        Args:
            notes (list): MIDI numbers or None for silence.
        """
        new = tuple(notes) if notes else ()
        if new == self.current:
            return
        for n in self.current:
            self.fs.noteoff(self.channel, n)
        for n in new:
            self.fs.noteon(self.channel, n, self.velocity)
        self.current = new

    def stop(self):
        """Stops current chord."""
        self.set_chord(None)