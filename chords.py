"""Music theory to turn hand gestures into chords."""

import pychord
from pychord import QualityManager



def register_qualities():
    """Add custom chord types to pychord's global quality table."""
    qm = QualityManager()
    qm.set_quality("dimM7",  ("1", "b3", "b5", "7"))
    qm.set_quality("M7sus4", ("1", "4", "5", "7"))
    qm.set_quality("M7sus2", ("1", "2", "5", "7"))
    qm.set_quality("7sus2",  ("1", "2", "5", "b7"))

NOTE_TO_SEMITONE = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}

def note_name_to_midi(name):
    """Convert a note name with octave to a MIDI note.
    
    Examples: 'C4' -> 60, 'Eb4' -> 63, 'F#3' -> 54
    
    Args:
        name (str): Letter, potential '#' or 'b' accidentals, then a
        single-digit octave.
    
    Return:
        int: MIDI note number
    """
    letter = name[0].upper()
    octave = int(name[-1])
    accidentals = name[1:-1]
    offset = accidentals.count("#") - accidentals.count("b")
    return 12 * (octave + 1) + NOTE_TO_SEMITONE[letter] + offset


class ChordHandler:
    """Holds current music state (chord quality)."""

    def __init__(self):
        """Initializes the module, sets default pychord parameters,
        and defines gesture keys to map to scale degrees and qualities."""
        self.note = 1       # C as default scale degree (1-7)
        self.chord_quality = "" 
        self.scale = "Cmaj" # C as default scale name
        # (index, middle, ring, pinky, thumb) --> base chord
        self.scale_degree = {
            (1, 0, 0, 0, 0): 1,   # pointing
            (1, 1, 0, 0, 0): 2,   # peace sign
            (1, 1, 1, 0, 0): 3,   # three up
            (1, 1, 1, 1, 0): 4,   # four up
            (1, 1, 1, 1, 1): 5,   # open palm
            (0, 0, 0, 0, 1): 6,   # thumb only
            (1, 0, 0, 0, 1): 7,   # thumb + index
        }
        # base chord --> triad quality in major key
        self.triad_quality = {1: "M", 2: "m", 3: "m", 4: "M", 5: "M", 6: "m", 7: "dim"}
        # (index, middle, ring, pinky) --> chord quality
        self.quality_gesture = {
            (1,0,0,0): {"M": "sus4",   "m": "sus4",   "dim": "sus4"},
            (0,1,0,0): {"M": "sus2",   "m": "sus2",   "dim": "sus2"},
            (1,1,0,0): {"M": "m",      "m": "M",      "dim": "m7b5"},
            (0,0,0,1): {"M": "maj7",   "m": "mM7",    "dim": "dimM7"},
            (0,0,1,1): {"M": "7",      "m": "m7",     "dim": "dim7"},
            (1,0,0,1): {"M": "M7sus4", "m": "M7sus4", "dim": "M7sus4"},
            (0,1,0,1): {"M": "M7sus2", "m": "M7sus2", "dim": "M7sus2"},
            (1,1,0,1): {"M": "mM7",    "m": "maj7",   "dim": "dimM7"},
            (1,0,1,1): {"M": "7sus4",  "m": "7sus4",  "dim": "7sus4"},
            (0,1,1,1): {"M": "7sus2",  "m": "7sus2",  "dim": "7sus2"},
            (1,1,1,1): {"M": "m7",     "m": "7",      "dim": "dim7"},
        }

    @property
    def base_quality(self):
        """Returns natural triad type ("M", "m", or "dim") for the
        current scale degree."""
        return self.triad_quality[self.note]

    def set_degree(self, key):
        """Looks up left hand pose in scale_degree and updates degree
        that pose is mapped to. Unmapped poses are ignored.
        
        Arg:
            key (tuple): 5-elements mapping to (index, middle, ring,
            pinky, thumb) defining whether or not each finger is
            extended.
        """
        chord = self.scale_degree.get(key
        )
        if chord is not None:                   #transitional poses ignored
            self.note = chord

    def set_quality(self, key):
        """Set the chord quality based on the right hand pose.

        Pose are split up into the thumb and the next four fingers.
        An extended thumb resets the quality to a plain diatonic triad.
        Otherwise, the other four fingers' states are looked up in 
        quality_gesture, using the current base_quality to pick the 
        right variant. Unmapped poses also result in the plain triad.
        
        Arg:
            key (tuple): 5-elements mapping to (index, middle, ring,
            pinky, thumb) defining whether or not each finger is
            extended.
        """
        *four, thumb = key
        if thumb: 
            self.chord_quality = ""
        else:
            self.chord_quality = self.quality_gesture.get(tuple(four), {}).get(self.base_quality, "")

    def resolve_quality(self):
        """Formats the gesture's chord qualities into a real pychord
        quality string. 
        
        Only needs to reformat default triad qualities and major
        qualities.
        
        Return:
            q (str): quality that pychord can parse
        """
        q = self.chord_quality
        if q == "":
            q = {"M": "", "m": "m", "dim": "dim"}.get(self.base_quality, "")
        elif q == "M":          # pychord spells major as ""
            q = ""
        return q

    def get_chord(self):
        """Build pychord chord.
        
        Return:
            pychord.Chord (tuple): 4 elements defining chord to store
        """
        quality = self.resolve_quality()
        # Some pychord versions spell half-diminished differently.
        candidates = [quality]
        if quality == "m7b5":
            candidates.append("m7-5")
        for q in candidates:
            try:
                return pychord.Chord.from_note_index(
                    note=self.note,
                    quality=q,
                    scale=self.scale,
                    diatonic=False,   # we pass the full quality ourselves
                )
            except Exception as e:
                print(f"GetChord failed for {q!r}: {e}")
                continue
        return None

    def current_notes(self):
        """Return chord name and midi note.
        
        Returns:
            str: chord name
            list: midi notes 
        """
        chord = self.get_chord()
        if chord is None:
            return "", []
        names = chord.components_with_pitch(root_pitch=4)   # e.g. ['C4','E4','G4']
        return str(chord), [note_name_to_midi(n) for n in names]