import cv2
import mediapipe as mp

# Initialize MediaPipe hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Initialize video capture
cap = cv2.VideoCapture('/home/nikolaus/cloud/rubato/dirigieren.mp4')  # Replace with your video file

with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break

        # Convert the image to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Process the image to find hand landmarks
        results = hands.process(image_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks on the image
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Access the right hand's index finger tip (assuming right hand is detected)
                for idx, hand in enumerate(results.multi_handedness):
                    if hand.classification[0].label == 'Left':
                        # Get the right index finger tip landmark (index 8)
                        right_index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                        #print(f"Right Index Finger Tip - x: {right_index_tip.x}, y: {right_index_tip.y}")

        # Display the image
        cv2.imshow("Right Index Finger Tracking", image)

        # Press ESC to exit
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
