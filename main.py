import time
import cv2 as cv
import mediapipe as mp

from looper import Loop
from audio import make_synth, ChordPlayer
from chords import ChordHandler, register_qualities
from gestures import StableValue, finger_key, FIST
from visuals import make_landmarkers, hand_side, to_pixels, draw_hand


GRACE = 0.3 # 0.3s to keep playing if left hand drops out briefly
prev = {"Left": None, "Right": None} # tracks prev frame's gesture for each hand

def main():
    # initialize webcam
    cam = cv.VideoCapture(0)
    if not cam.isOpened():
        print("Could not open camera")
        return

    # initialize musical elements
    register_qualities()
    synth = make_synth()
    player = ChordPlayer(synth) # real-time playback
    loop_player = ChordPlayer(synth, channel=1) # background looping
    handler = ChordHandler()
    loop = Loop()

    # smooths sudden/random gestures over 2 frames
    stabilizer = {"Left": StableValue(2), "Right": StableValue(2)}
    
    # track last time left hand visible
    last_left = -1e9 
    start_time = time.monotonic()

    try: 
        # open mediapipe hand tracking model
        with make_landmarkers() as landmarker:
            while cam.isOpened():
                success, frame = cam.read()
                if not success:
                    print("Camera frame not available")
                    break

                # flip frame for mirror view
                frame = cv.flip(frame, 1)
                h, w = frame.shape[:2]

                # mediapipe reads RGB, opencv reads BGR
                rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                
                timestamp_ms = int((time.monotonic() - start_time) * 1000)
                result = landmarker.detect_for_video(mp_image, timestamp_ms)

                # loop through every detected hand
                seen = set()
                for landmarks, handedness in zip(result.hand_landmarks, result.handedness):
                    side = hand_side(handedness)
                    seen.add(side)

                    # Convert normalized coords to pixels so distances are not
                    # skewed by the frame's aspect ratio.
                    pts = to_pixels(landmarks, w, h)

                    # identify gesture based on finger positions, then smooth
                    key = stabilizer[side].update(finger_key(pts))

                    # left hand controls scale degree
                    # right hand controls chord quality & loop status
                    just_fisted = (key == FIST and prev[side] != FIST)
                    prev[side] = key

                    if key is not None:
                        if side == "Left":
                            handler.set_degree(key)

                        else:
                            if just_fisted:
                                state = loop.on_fist()
                                print(f"FIST fired --> {state}")
                            handler.set_quality(key)
                    # visualize landmarks and detected gesture
                    draw_hand(frame, pts, f"{side}: {key}")

                # update status for hands not in view
                for side in prev: 
                    if side not in seen:
                        prev[side] = None
                        stabilizer[side] = StableValue(2)

                
                now = time.monotonic()
                if "Left" in seen:
                    last_left = now

                name, notes = "-", None

                # grace period for left hand leaving screen (0.3s)
                # previous chord data still held
                if now - last_left <= GRACE:
                    name, notes = handler.current_notes()
                live = tuple(notes) if notes else ()

                # loop states
                if loop.state == "recording":
                    loop.save_chords(live)

                player.set_chord(live) # live hands always play sound

                if loop.state == "playing":
                    loop_player.set_chord(loop.playback())
                else:
                    loop_player.set_chord(()) # silence loop when not playing

                # render current chord names on upper left screen
                cv.putText(frame, f"Chord: {name or '-'}", (10, 30),
                        cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
                # open window 
                cv.imshow("Hand Synthesizer", frame)
                # exit program if "q"
                if cv.waitKey(1) & 0xFF == ord("q"):
                    break

    finally: 
        player.stop()
        loop_player.stop()
        synth.delete()
        cam.release()
        cv.destroyAllWindows()


if __name__ == "__main__":
    main()