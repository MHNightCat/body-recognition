import cv2
import mediapipe as mp
from PIL import ImageGrab
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

screen_width, screen_height = ImageGrab.grab().size

WINDOW_WIDTH, WINDOW_HEIGHT = 1000, 500
scale_x = WINDOW_WIDTH / screen_width
scale_y = WINDOW_HEIGHT / screen_height

mouse_x, mouse_y = 0, 0

with mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as pose:
  while True:
    image = ImageGrab.grab()

    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image)
    
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    mp_drawing.draw_landmarks(
        image,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
    
    height, width, _ = image.shape
    scale = min(WINDOW_WIDTH/width, WINDOW_HEIGHT/height)
    image = cv2.resize(image, (0, 0), fx=scale, fy=scale)

    cv2.imshow('MediaPipe Pose', image)
    
    if cv2.waitKey(1) == ord('q'):
        break

cv2.destroyAllWindows()