import os
import cv2
import urllib.request
import numpy as np
from utils.logger import get_logger
from config.settings import HAND_LANDMARKER_PATH

logger = get_logger("mediapipe_shim")

# Standard hand landmark connections
HAND_CONNECTIONS = [
    # Thumb
    (0, 1), (1, 2), (2, 3), (3, 4),
    # Index
    (0, 5), (5, 6), (6, 7), (7, 8),
    # Middle
    (9, 10), (10, 11), (11, 12),
    # Ring
    (13, 14), (14, 15), (15, 16),
    # Pinky
    (0, 17), (17, 18), (18, 19), (19, 20),
    # Palm knuckles
    (5, 9), (9, 13), (13, 17)
]

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
MODEL_PATH = HAND_LANDMARKER_PATH

def download_model_if_needed():
    """Programmatically downloads the official Google Hand Landmarker model if missing."""
    if not os.path.exists(MODEL_PATH):
        logger.warning(f"'{MODEL_PATH}' not found locally.")
        logger.info(f"Downloading pre-trained Hand Landmarker model from Google ({MODEL_URL})...")
        try:
            import ssl
            # Bypass SSL certificate verification issues common on macOS
            ssl_context = ssl._create_unverified_context()
            
            # Create custom opener that uses the unverified context
            opener = urllib.request.build_opener(urllib.request.HTTPSHandler(context=ssl_context))
            opener.addheaders = [('User-Agent', 'Mozilla/5.0')]
            urllib.request.install_opener(opener)
            
            # Ensure the directory exists
            os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
            
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
            logger.info("Model downloaded successfully!")
        except Exception as e:
            logger.error(f"Error downloading model: {e}")
            raise e

class ShimLandmark:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

class ShimHandLandmarks:
    def __init__(self, landmarks):
        # Convert a list of normalized keypoint landmarks
        self.landmark = [ShimLandmark(lm.x, lm.y, lm.z) for lm in landmarks]

class ShimResults:
    def __init__(self, multi_hand_landmarks):
        self.multi_hand_landmarks = multi_hand_landmarks

class ShimHands:
    """
    A drop-in wrapper around modern Google MediaPipe Tasks HandLandmarker 
    that exposes the exact same interface as the legacy mp.solutions.hands.Hands.
    """
    def __init__(self, static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7):
        download_model_if_needed()
        
        # Lazy load to avoid import errors on initialization
        from mediapipe.tasks.python import vision
        from mediapipe.tasks.python.core import base_options
        
        self.options = vision.HandLandmarkerOptions(
            base_options=base_options.BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=vision.RunningMode.IMAGE,
            num_hands=max_num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.landmarker = vision.HandLandmarker.create_from_options(self.options)
        
    def process(self, rgb_frame):
        """Processes an RGB frame and returns landmarks in legacy format."""
        from mediapipe import Image, ImageFormat
        
        # Convert frame to MediaPipe Image object
        mp_image = Image(image_format=ImageFormat.SRGB, data=rgb_frame)
        
        # Run inference
        result = self.landmarker.detect(mp_image)
        
        # Translate to legacy solutions results shape
        if result and result.hand_landmarks:
            multi_hand_landmarks = [ShimHandLandmarks(hand) for hand in result.hand_landmarks]
            return ShimResults(multi_hand_landmarks)
            
        return ShimResults(None)
        
    def close(self):
        if hasattr(self, 'landmarker'):
            self.landmarker.close()
            
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

# Alias for legacy solutions compatibility
Hands = ShimHands

def draw_custom_landmarks(frame, hand_landmarks):
    """
    Draws a premium, high-tech glowing hand skeleton using OpenCV.
    Replaces the missing MediaPipe Solutions drawing utilities.
    """
    h, w, _ = frame.shape
    
    # 1. Draw connections (Glowing Neon Yellow/Green skeleton line)
    for connection in HAND_CONNECTIONS:
        start_idx, end_idx = connection
        if start_idx < len(hand_landmarks.landmark) and end_idx < len(hand_landmarks.landmark):
            lm_start = hand_landmarks.landmark[start_idx]
            lm_end = hand_landmarks.landmark[end_idx]
            
            pt_start = (int(lm_start.x * w), int(lm_start.y * h))
            pt_end = (int(lm_end.x * w), int(lm_end.y * h))
            
            # Thick background soft shadow line (cyan glow)
            cv2.line(frame, pt_start, pt_end, (255, 255, 0), 3, cv2.LINE_AA)
            # Thin sharp inner line (white cores)
            cv2.line(frame, pt_start, pt_end, (255, 255, 255), 1, cv2.LINE_AA)
            
    # 2. Draw key joint coordinates (Neon Red dots)
    for idx, lm in enumerate(hand_landmarks.landmark):
        pt = (int(lm.x * w), int(lm.y * h))
        
        # Highlight index tip (landmark 8) with a special gold target ring
        if idx == 8:
            cv2.circle(frame, pt, 8, (0, 215, 255), 2, cv2.LINE_AA)
            cv2.circle(frame, pt, 4, (0, 215, 255), -1, cv2.LINE_AA)
        else:
            # Regular joints: red dot with white outline
            cv2.circle(frame, pt, 5, (0, 0, 255), -1, cv2.LINE_AA)
            cv2.circle(frame, pt, 6, (255, 255, 255), 1, cv2.LINE_AA)
