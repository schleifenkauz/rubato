from cv2 import VideoCapture
from datetime import datetime

def get_video_source():
    return VideoCapture(0) # returns default video input (e.g. Webcam)
    # To use a video file as the input use
    # return VideoCapture("path/to/the/video")

def get_output_file_name():
    # Uncomment to record video to disk while analyzing
    #timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    #return f"data/video{timestamp}.avi"
    return None
