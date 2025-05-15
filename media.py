import cv2
import mediapipe as mp
from mediapipe.python.solutions.drawing_utils import DrawingSpec
from local_config import get_video_source
import numpy as np
from util import calculate_average_point

def draw_circle(target, point, radius, color):
    h, w, _ = target.shape
    px = int(point[0] * w)
    py = int(point[1] * h)

    cv2.circle(target, (px, py), radius, color, thickness=-1)


mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils  # Add drawing utilities
pose = mp_pose.Pose(
    static_image_mode=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def analyze_video(analyze_frame, show_video):
    cap = get_video_source()
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Starting to analyze video. FPS: {fps}")
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Error reading video stream.")
            break
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)

        if show_video:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            overlay = np.zeros(image.shape, dtype=np.uint8)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks
            right_hand = calculate_average_point(landmarks.landmark[16:22:2])
            analyze_frame(right_hand, fps)
            if show_video:
                spec = DrawingSpec((20, 20, 20), -1, 3)
                mp_drawing.draw_landmarks(overlay, landmarks, mp_pose.POSE_CONNECTIONS, spec)
                draw_circle(overlay, right_hand, 5, color=(0, 255, 0))

        if show_video:
            image = cv2.addWeighted(overlay, 1, image, 0.1, 0)
            cv2.imshow("Live", image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
