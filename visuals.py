"""Hand tracking setup and drawing helpers for popup display."""

import os
import urllib.request
import cv2 as cv
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision


# Setup and drawing
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")

# pairs of landmark indices
# joined by line when hand drawn
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]

def ensure_model():
    """Download hand landmarker model."""
    if not os.path.exists(MODEL_PATH):
        print("Downloading hand landmarker model (one time)...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

def draw_hand(frame, pts, label):
    """Draw a hand skeleton and text label on a given frame.
    
    Args:
        frame (numpy.ndarray): BGR image to draw on
        pts (list): 21 (x,y) pixel coordinates (one per landmark)
        label (str): text drawn below wrist representing current finger key
    """
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


def make_landmarkers():
    """Create a MediaPipe hand landmarker for processing video frames.
    
    Return:
        vision.HandLandmarker: Landmarker in VIDEO mode"""
    ensure_model()
    options = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return vision.HandLandmarker.create_from_options(options)

def hand_side(handedness):
    """Return which hand was detected and correct for the mirror-view.
    
    Arg:
        handedness (list): Mediapipe's handedness categories for one hand, 
        first entry holds predicted label
        
    Return:
        str: "Right" or "Left" from user's POV
    """
    return "Right" if handedness[0].category_name == "Left" else "Left"

def to_pixels(landmarks, w, h):
    """Convert normalized landmarks to pixel coordinates.
    
    Args:
        landmarks (list): Mediapipe landmarks with normalized x and y.
        w (int): frame width in pixels
        h (int): frame height in pixels
    
    Return:
        list: (x,y) pixel tuples for each landmark
    """
    return [(lm.x * w, lm.y * h) for lm in landmarks]