import math
import os
import time
import fluidsynth
import urllib.request
import pychord
from collections import deque

import cv2 as cv
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from pychord import QualityManager

qm = QualityManager()
qm.set_quality("dimM7",  ("1", "b3", "b5", "7"))
qm.set_quality("M7sus4", ("1", "4", "5", "7"))
qm.set_quality("M7sus2", ("1", "2", "5", "7"))
qm.set_quality("7sus2",  ("1", "2", "5", "b7"))

# music
fs = fluidsynth.Synth()
fs.start(driver="pulseaudio")   # if this fails, try "alsa"
sfid = fs.sfload("/usr/share/sounds/sf2/FluidR3_GM.sf2")
fs.program_select(0, sfid, 0, 48)


MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]

# (tip index, middle-joint index) for each non-thumb finger.
FINGER_JOINTS = {
    "index": (8, 6),
    "middle": (12, 10),
    "ring": (16, 14),
    "pinky": (20, 18),
}

NOTE_TO_SEMITONE = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}


def note_name_to_midi(name):
    """'C4' -> 60, 'Eb4' -> 63, 'F#3' -> 54."""
    letter = name[0].upper()
    octave = int(name[-1])
    accidentals = name[1:-1]
    offset = accidentals.count("#") - accidentals.count("b")
    return 12 * (octave + 1) + NOTE_TO_SEMITONE[letter] + offset

# ---------------------------------------------------------------------------
# Gesture logic
# ---------------------------------------------------------------------------

def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def finger_states(pts):
    """Return {finger_name: True if extended}. pts is a list of 21 (x, y) pixel points."""
    wrist = pts[0]
    states = {}

    # Non-thumb fingers: extended if the tip is farther from the wrist than the
    # middle joint is. Using distance (not "tip above joint") means it works
    # even when the hand is rotated or sideways.
    for name, (tip, joint) in FINGER_JOINTS.items():
        states[name] = dist(pts[tip], wrist) > dist(pts[joint], wrist)

    # Thumb: it moves sideways, so compare against the base of the pinky (17).
    # When tucked across the palm the tip is closer to that point than the
    # thumb's own knuckle (3) is; when extended it's farther away.
    states["thumb"] = dist(pts[4], pts[17]) > dist(pts[3], pts[17])
    return states


def classify(pts):
    """Return (gesture_name, number_of_extended_fingers)."""
    s = finger_states(pts)
    count = sum(s.values())

    # Hand size (wrist to middle-finger base) lets us measure distances
    # relative to the hand, so results don't change as you move closer/farther.
    hand_size = dist(pts[0], pts[9])
    pinching = dist(pts[4], pts[8]) / hand_size < 0.25

    four_up = [s["index"], s["middle"], s["ring"], s["pinky"]]

    if pinching and s["middle"] and s["ring"] and s["pinky"]:
        return "OK", count

    if count == 5:
        return "Open palm", count

    if not any(four_up):
        if not s["thumb"]:
            return "Fist", count
        # Only the thumb is out: up, down, or sideways?
        if pts[4][1] < pts[0][1] - 0.5 * hand_size:   # image y grows downward
            return "Thumbs up", count
        if pts[4][1] > pts[0][1] + 0.5 * hand_size:
            return "Thumbs down", count
        return "Thumb out", count

    if s["index"] and not any([s["middle"], s["ring"], s["pinky"]]) and not s["thumb"]:
        return "Pointing", count

    if s["index"] and s["middle"] and not s["ring"] and not s["pinky"] and not s["thumb"]:
        return "Peace", count

    return f"{count} fingers", count

class MusicElement:
    """one sound to play"""

    def __init__(self):
        self.pitch = 60; # middle c

    def play(self, gesture):
        # play the sound

        if gesture == "Pointing":
            fs.noteon(0, self.pitch, 100)
        else:
            fs.noteoff(0, self.pitch)

    def stop(self):
        # stop current sound
        fs.noeoff(0, self.pitch)

class ChordHandler:

    def __init__(self):
        self.left_hand = None
        self.right_hand = None
        self.note = 1 # C as default root note
        self.chord_quality = "" 
        self.scale = "Cmaj" # C as default scale
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
        # (index, middle, ring, pinky, thumb) --> chord quality
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
        return self.triad_quality[self.note]

    def chord_gestures(self, pts, side):
        """Left hand: choose scale degree. Right hand: modify chord quality."""
        s = finger_states(pts)

        if side == "Left":
            key = tuple(int(s[f]) for f in ("index", "middle", "ring", "pinky", "thumb"))
            chord = self.scale_degree.get(key)
            if chord is not None:
                self.note = chord
                self.left_hand = f"SD{chord}"

        if side == "Right":
            key = tuple(int(s[f]) for f in ("index", "middle", "ring", "pinky"))
            if s["thumb"]:
                self.chord_quality = ""
            else:
                self.chord_quality = self.quality_gesture.get(key, {}).get(self.base_quality, "")

        return (self.note, self.chord_quality)

    def resolve_quality(self):
        """Turn the gesture's quality into a real pychord quality string."""
        q = self.chord_quality
        if q == "":
            q = {"M": "", "m": "m", "dim": "dim"}.get(self.base_quality, "")
        elif q == "M":          # pychord spells major as ""
            q = ""
        return q

    def get_chord(self):
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

    def get_midi_notes(self):
        chord = self.get_chord()
        if chord is None:
            return None
        names = chord.components_with_pitch(root_pitch=4)   # e.g. ['C4','E4','G4']
        return [note_name_to_midi(n) for n in names]

class ChordPlayer:
    """plays one chord at a time"""
    def __init__(self, channel=0, velocity=90):
        self.channel = channel
        self.velocity = velocity
        self.current = ()          # tuple of MIDI notes currently sounding

    def set_chord(self, notes):
        """notes: list of MIDI numbers, or None/[] for silence."""
        new = tuple(notes) if notes else ()
        if new == self.current:
            return
        for n in self.current:
            fs.noteoff(self.channel, n)
        for n in new:
            fs.noteon(self.channel, n, self.velocity)
        self.current = new

    def stop(self):
        self.set_chord(None)


class GestureSmoother:
    """Only report a gesture once it has been the same for N frames in a row."""

    def __init__(self, frames=5):
        self.frames = frames
        self.history = {}
        self.stable = {}

    def update(self, key, gesture):
        h = self.history.setdefault(key, deque(maxlen=self.frames))
        h.append(gesture)
        if len(h) == self.frames and len(set(h)) == 1:
            if self.stable.get(key) != gesture:
                pass
                #print(f"{key} hand: {gesture}")   # <-- hook point for real actions
            self.stable[key] = gesture
        return self.stable.get(key, "...")

    def forget_missing(self, seen_keys):
        for key in list(self.history):
            if key not in seen_keys:
                del self.history[key]
                self.stable.pop(key, None)


# Setup and drawing

def ensure_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading hand landmarker model (one time)...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)


def draw_hand(frame, pts, label, gesture):
    int_pts = [(int(x), int(y)) for x, y in pts]
    for start, end in HAND_CONNECTIONS:
        cv.line(frame, int_pts[start], int_pts[end], (255, 255, 255), 2)

    color = (255, 255, 255)
    # if gesture == "Thumbs up":
    #     color = (0, 200, 0)
    # elif gesture == "Thumbs down":
    #     color = (0, 0, 200)
    # elif gesture == "Pointing":
    #     color = (200, 0, 0)

    for p in int_pts:
        cv.circle(frame, p, 4, color, -1)
    cv.putText(frame, label, (int_pts[0][0], int_pts[0][1] + 25),
               cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)


def main():
    ensure_model()

    options = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cam = cv.VideoCapture(0)
    if not cam.isOpened():
        print("Could not open camera")
        return

    handler = ChordHandler()
    player = ChordPlayer()
    chord_history = deque(maxlen=5)
    smoother = GestureSmoother(frames=5)
    start_time = time.monotonic()
    stable_chord = ("", ())

    with vision.HandLandmarker.create_from_options(options) as landmarker:
        while cam.isOpened():
            success, frame = cam.read()
            if not success:
                print("Camera frame not available")
                break

            frame = cv.flip(frame, 1)
            h, w = frame.shape[:2]

            rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int((time.monotonic() - start_time) * 1000)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            seen = set()
            total_fingers = 0

            for landmarks, handedness in zip(result.hand_landmarks, result.handedness):
                raw = handedness[0].category_name
                side = "Right" if raw == "Left" else "Left"   # Tasks API labels are inverted for a flipped frame
                seen.add(side)

                # Convert normalized coords to pixels so distances are not
                # skewed by the frame's aspect ratio.
                pts = [(lm.x * w, lm.y * h) for lm in landmarks]

                gesture, count = classify(pts)
                total_fingers += count
                stable_gesture = smoother.update(side, gesture)

                chord_disp = handler.chord_gestures(pts, side)
                draw_hand(frame, pts, f"{side}: {chord_disp}", stable_gesture)
                

            smoother.forget_missing(seen)

            # Desired chord: only sounds while the left hand (scale degree) is visible.
            chord = handler.get_chord() if "Left" in seen else None
            if chord is not None:
                notes = tuple(note_name_to_midi(n)
                              for n in chord.components_with_pitch(root_pitch=4))
                desired = (str(chord), notes)
            else:
                desired = ("", ())

            chord_history.append(desired)
            if len(chord_history) == chord_history.maxlen and len(set(chord_history)) == 1:
                stable_chord = desired
            player.set_chord(stable_chord[1])

            chord_name = stable_chord[0] or "-"
            cv.putText(frame, f"Chord: {chord_name}", (10, 30),
                       cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
            cv.imshow("Gestures", frame)
            if cv.waitKey(1) & 0xFF == ord("q"):
                break

    player.stop()
    fs.delete()
    cam.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()