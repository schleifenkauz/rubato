import cv2
import mediapipe as mp
from mediapipe.python.solutions.drawing_utils import DrawingSpec
from local_config import get_video_source, get_output_file_name
from util import calculate_average_point
from gestures import is_fist


def draw_circle(target, point, radius, color):
    h, w, _ = target.shape
    px = int(point[0] * w)
    py = int(point[1] * h)

    cv2.circle(target, (px, py), radius, color, thickness=-1)


mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils  # Add drawing utilities

# Analyze a video with media-pipe and feed the results into [analyze_frame].
# [option] can be either "hands" or "pose". This decides which media-pipe model is used for the analysis.
# If [show_video] is True the each frame is shown on the screen with the detected landmarks overlayed.
#
# This function probably doesn't have to be modified. It acts only as a wrapper around the passed [analyze_frame] function which does the real work. 
def analyze_video(option, complexity, analyze_frame, show_video):
    if option == "hands":
        hands = mp_hands.Hands(
            static_image_mode=True,
            min_detection_confidence=0.2,
            model_complexity=complexity,
            min_tracking_confidence=0.2
        )
    elif option == "pose":
        pose = mp_pose.Pose(
            static_image_mode=False,
            min_detection_confidence=0.5,
            model_complexity=complexity,
            min_tracking_confidence=0.5
        )
    else:
        print(f"Invalid option '{option}'. Must be either 'hands' or 'pose'.")
        return

    cap = get_video_source()
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Starting to analyze video. [{width} x {height}] FPS: {fps}")

    output_filename = get_output_file_name()

    if output_filename:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Error reading video stream.")
            break
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if output_filename:
            out.write(image)

        if option == "hands":
            hand_results = hands.process(image)
            if hand_results.multi_hand_landmarks and hand_results.multi_handedness:
                for i, hand_landmarks in enumerate(hand_results.multi_hand_landmarks):
                    if hand_results.multi_handedness[i].classification[0].label == "Left":
                        right_hand = calculate_average_point(hand_landmarks.landmark)
                        analyze_frame(right_hand, [0, 0], fps)
                        if show_video:
                            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            else:
                cv2.putText(image, "No hand detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        if option == "pose":
            results = pose.process(image)
            if results.pose_landmarks:
                landmarks = results.pose_landmarks
                right_hand = calculate_average_point(landmarks.landmark[16:23:2])
                left_hand = calculate_average_point(landmarks.landmark[15:22:2])
                analyze_frame(right_hand, left_hand, fps)

                if show_video:
                    spec = DrawingSpec((20, 20, 20), -1, 3)
                    mp_drawing.draw_landmarks(image, landmarks, mp_pose.POSE_CONNECTIONS, spec)
                    draw_circle(image, right_hand, 5, color=(0, 255, 0))

        if show_video:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imshow("Live", image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    if output_filename:
        out.release()
    cv2.destroyAllWindows()
