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
pose = mp_pose.Pose(
    static_image_mode=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
hands = mp_hands.Hands(
    static_image_mode=True,
    min_detection_confidence=0.2,
    min_tracking_confidence=0.2
)

def analyze_video(option, analyze_frame, show_video):
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

        
        hand_results = hands.process(image)
        is_hand_fist = False
        if hand_results.multi_hand_landmarks and hand_results.multi_handedness:
            for i, hand_landmarks in enumerate(hand_results.multi_hand_landmarks):
                if hand_results.multi_handedness[i].classification[0].label == "Left":
                    is_hand_fist = False #is_fist(hand_landmarks)
                    message = f"Hand is {'fist' if is_hand_fist else 'open'}"
                    mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    #cv2.putText(image, message, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                    if option == "hands" and not is_hand_fist:
                        right_hand = calculate_average_point(hand_landmarks.landmark)
                        analyze_frame(right_hand, [0, 0], fps)
        else :
            None
            #cv2.putText(image, "No hand detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        if option == "pose" and not is_hand_fist:
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
