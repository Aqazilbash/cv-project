"""Gesture logic to determine extension of fingers for future 
classification and to smooth gestures between frames."""

import math



# (tip index, middle-joint index) for each non-thumb finger.
FINGER_JOINTS = {
    "index": (8, 6),
    "middle": (12, 10),
    "ring": (16, 14),
    "pinky": (20, 18),
}

FINGER_ORDER = ("index", "middle", "ring", "pinky", "thumb")

FIST = (0,0,0,0,0)

def dist(a, b):
    """Determine distance between two points.
    
    Args:
        a (tuple): x and y coordinates of first finger joint
        b (tuple): x and y coordinates finger joint
    
    Return:
        float: distance in pixels
    """
    return math.hypot(a[0] - b[0], a[1] - b[1])

def finger_key(pts):
    """Create key defining whether or not a finger is extended for 
    all five fingers.
    
    Arg:
        pts (list): 21 (x,y) pixel coordinates for each finger joint
    
    Return: 
        tuple: 5 elements of 0s and 1s for each finger
    """
    s = finger_states(pts)
    return tuple(int(s[f]) for f in FINGER_ORDER)

def finger_states(pts):
    """Determine if finger extended. 
    
    If the distance between the fingertip and wrist are shorter than the 
    middle joint and wrist, the finger is not extended. For the thumb, 
    compare the fingertip and the base joint of the pinky.

    Arg: 
        pts (list): 21 (x,y) pixel coordinates for each finger joint

    Return: 
        states (dict): 5 elements with finger name as the key and a boolean 
        for extension as the value.
    """
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


class StableValue():
    """Switch finger tuple to a new value only after it's been seen several
    times in a row."""
    def __init__(self, confirm=2):
        """Initialize the smoother.
        
        Arg:
            confirm (int): consecutive frames needed to accept a change."""
        self.confirm = confirm
        self.value = None
        self.pending = None
        self.count = 0

    def update(self, v): 
        """Feed the newest raw value and return a stable one.
        
        Arg:
            v (tuple): five 0 and 1 values for the raw finger key of
            a given frame

        Return:
            tuple or None: the confirmed finger key or None if no
            value has been confirmed yet
        """
        if v == self.value:
            self.pending, self.count = None, 0
        elif v == self.pending: 
            self.count += 1
            if self.count >= self.confirm:
                self.value, self.pending, self.count = v, None, 0
        else: 
            self.pending, self.count = v, 1
            if self.confirm <= 1:
                self.value, self.pending, self.count = v, None, 0
        return self.value