import cv2
import numpy as np
import os
import csv
import sys

# Dynamically append the project root directory to sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from tracking.normalization import normalize_landmarks
from utils.logger import get_logger
from config.settings import CLASSES, DATASET_PATH

logger = get_logger("collect_data")

# Try standard MediaPipe, fallback to custom tasks shim if solutions unavailable (e.g. Python 3.13)
try:
    import mediapipe.solutions.hands as mp_hands
    import mediapipe.solutions.drawing_utils as mp_drawing
    import mediapipe.solutions.drawing_styles as mp_drawing_styles
    USE_SHIM = False
except (ModuleNotFoundError, AttributeError):
    import utils.mediapipe_shim as mp_hands
    from utils.mediapipe_shim import draw_custom_landmarks as mp_drawing
    USE_SHIM = True


def draw_hud(frame, current_class, is_recording, counts):
    """
    Draws a premium, modern glassmorphic HUD overlay with translucent backgrounds and clean typography.
    """
    height, width, _ = frame.shape

    # Create semi-transparent overlay for the control panel (glassmorphism effect)
    overlay = frame.copy()

    # Stat panel background (height adjusted for 8 classes)
    cv2.rectangle(overlay, (15, 15), (320, 310), (30, 30, 30), -1)

    # Active Zone boundary guide (central 60% rectangle)
    az_x_min, az_x_max = int(width * 0.2), int(width * 0.8)
    az_y_min, az_y_max = int(height * 0.2), int(height * 0.8)
    cv2.rectangle(frame, (az_x_min, az_y_min), (az_x_max, az_y_max), (150, 150, 150), 1)
    cv2.putText(
        frame,
        "ACTIVE TRACKING ZONE",
        (az_x_min + 5, az_y_min - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (150, 150, 150),
        1,
        cv2.LINE_AA,
    )

    # Blend overlay with original frame (40% opacity for HUD panel)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    # HUD Title
    cv2.putText(
        frame,
        "GESTURE DATA COLLECTOR",
        (25, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.line(frame, (25, 48), (305, 48), (100, 100, 100), 1)

    # Recording Status Badge
    status_text = "RECORDING" if is_recording else "PAUSED"
    status_color = (0, 0, 255) if is_recording else (0, 255, 0)
    cv2.rectangle(frame, (25, 60), (140, 85), status_color, -1)
    cv2.putText(
        frame,
        status_text,
        (35, 78),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 0) if not is_recording else (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    # Current Class label
    cv2.putText(
        frame,
        "Active State:",
        (25, 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"{CLASSES[current_class].upper()} ({current_class})",
        (135, 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 180, 50),
        2,
        cv2.LINE_AA,
    )

    # Class sample counts table
    cv2.putText(
        frame,
        "Collected samples:",
        (25, 135),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (150, 150, 150),
        1,
        cv2.LINE_AA,
    )

    y_offset = 158
    for code, name in CLASSES.items():
        # Highlight active class in the counts table
        color = (255, 255, 255) if code == current_class else (180, 180, 180)
        thickness = 2 if code == current_class else 1

        cv2.putText(
            frame,
            f"[{code}] {name.capitalize()}:",
            (35, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            thickness,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"{counts.get(code, 0)}",
            (220, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (100, 255, 100) if counts.get(code, 0) > 0 else (100, 100, 100),
            2,
            cv2.LINE_AA,
        )
        y_offset += 18

    # Instructions Overlay at bottom right
    inst_overlay = frame.copy()
    cv2.rectangle(
        inst_overlay,
        (width - 290, height - 120),
        (width - 15, height - 15),
        (20, 20, 20),
        -1,
    )
    cv2.addWeighted(inst_overlay, 0.7, frame, 0.3, 0, frame)

    cv2.putText(
        frame,
        "HOTKEYS:",
        (width - 280, height - 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        "[0-7] Switch gesture state",
        (width - 280, height - 82),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        "[Space] Start/Pause recording",
        (width - 280, height - 67),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        "[C] Clear selected class dataset",
        (width - 280, height - 52),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        "[Q] Save and Quit",
        (width - 280, height - 37),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (100, 150, 255),
        1,
        cv2.LINE_AA,
    )


def load_existing_counts(csv_path):
    """Loads existing dataset sample counts from the CSV to preserve progress."""
    counts = {code: 0 for code in CLASSES.keys()}
    if not os.path.exists(csv_path):
        return counts

    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) > 0:
                try:
                    label = int(row[0])
                    if label in counts:
                        counts[label] += 1
                except ValueError:
                    # Skip header if present
                    continue
    return counts


def main():
    csv_path = DATASET_PATH

    # Load counts of already collected data
    counts = load_existing_counts(csv_path)

    # In-memory buffer to hold samples collected during this session
    session_data = []

    # Initialize MediaPipe Hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
    )

    # Open camera stream
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("Could not access the webcam.")
        return

    # Set frame dimensions to 1280x720 (HD) for crystal clear display
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    current_class = 0
    is_recording = False

    logger.info("Hand Gesture Data Collection Started")
    logger.info(f"Dataset destination: {os.path.abspath(csv_path)}")
    logger.info("Press Q inside the camera window to save and exit.")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            logger.warning("Ignoring empty camera frame.")
            continue

        # Flip the frame horizontally for a natural mirror-like viewing experience
        frame = cv2.flip(frame, 1)

        # Convert BGR image to RGB for MediaPipe processing
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Run hand landmarks detection
        results = hands.process(rgb_frame)

        has_hand = False
        normalized_feats = None

        # If hand detected
        if results.multi_hand_landmarks:
            has_hand = True
            hand_landmarks = results.multi_hand_landmarks[0]

            # Draw MediaPipe hand connection mesh beautifully
            if USE_SHIM:
                mp_drawing(frame, hand_landmarks)
            else:
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style(),
                )

            # Normalize landmarks using translation/scale invariant algorithm
            normalized_feats = normalize_landmarks(hand_landmarks)

            # Record if recording is active
            if is_recording and normalized_feats is not None:
                session_data.append([current_class] + normalized_feats)
                counts[current_class] += 1

        # Draw glassmorphic HUD
        draw_hud(frame, current_class, is_recording, counts)

        # Render visual warning if recording but no hand is in frame
        if is_recording and not has_hand:
            cv2.putText(
                frame,
                "WARNING: NO HAND DETECTED",
                (400, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

        # Show output window
        cv2.imshow("Hand Gesture Mouse - Data Collector", frame)

        # Capture keyboard events
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q") or key == ord("Q"):
            break
        elif key == ord(" "):  # Spacebar toggles recording
            is_recording = not is_recording
            logger.info(f"Recording state: {'ON' if is_recording else 'OFF'}")
        elif ord("0") <= key <= ord("7"):  # 0 to 7 sets the active class
            current_class = key - ord("0")
            logger.info(
                f"Switched active recording class to: {CLASSES[current_class].upper()} ({current_class})"
            )
        elif key == ord("c") or key == ord("C"):
            # Clear data for selected class
            logger.info(
                f"Clearing collected data for class {CLASSES[current_class].upper()} in this session."
            )
            session_data = [row for row in session_data if row[0] != current_class]
            # Reload counts from CSV to show correct total
            temp_counts = load_existing_counts(csv_path)
            counts[current_class] = temp_counts[current_class] + sum(
                1 for row in session_data if row[0] == current_class
            )

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

    # Save session data to CSV on exit
    if session_data:
        logger.info(f"Saving {len(session_data)} new samples to {csv_path}...")

        # Append data to the CSV file
        os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
        file_exists = os.path.exists(csv_path)
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            # Write header if new file
            if not file_exists:
                header = ["label"] + [f"feat_{i}" for i in range(63)]
                writer.writerow(header)
            writer.writerows(session_data)

        logger.info("Data saved successfully!")
    else:
        logger.info("No new samples collected in this session.")

    logger.info("Goodbye!")


if __name__ == "__main__":
    main()
