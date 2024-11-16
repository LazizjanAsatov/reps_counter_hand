import cv2
import mediapipe as mp
import numpy as np
mp_drawing=mp.solutions.drawing_utils
mp_pose=mp.solutions.pose

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180.0 else angle

POSE_CONFIG = {
    'min_detection_confidence': 0.5,
    'min_tracking_confidence': 0.5
}
FONT = cv2.FONT_HERSHEY_SIMPLEX
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
ORANGE = (245, 117, 16)
RED = (0, 0, 255)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
counter = 0
stage = None

with mp_pose.Pose(**POSE_CONFIG) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        try:
            landmarks = results.pose_landmarks.landmark
            h, w, _ = image.shape
            
            keypoints = {
                'shoulder_left': [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                           landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y],
                'elbow_left': [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y],
                'wrist_left': [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y],
                'shoulder_right': [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                           landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y],
                'elbow_right': [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                         landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y],
                'wrist_right': [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                         landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y],
            }

            angle_left = calculate_angle(keypoints['shoulder_left'], keypoints['elbow_left'], keypoints['wrist_left'])
            angle_right = calculate_angle(keypoints['shoulder_right'], keypoints['elbow_right'], keypoints['wrist_right'])
            
            elbow_px_left = tuple(np.multiply(keypoints['elbow_left'], [w, h]).astype(int))
            elbow_px_right = tuple(np.multiply(keypoints['elbow_right'], [w, h]).astype(int))
            cv2.putText(image, f"{int(angle_left)}°", elbow_px_left, FONT, 0.5, WHITE, 2, cv2.LINE_AA)
            cv2.putText(image, f"{int(angle_right)}°", elbow_px_right, FONT, 0.5, WHITE, 2, cv2.LINE_AA)
            if angle_left > 160 and angle_right > 160:
                stage = "down"
            elif (angle_left < 30 and angle_right < 30) and stage == "down":
                stage = "up"
                counter += 1
                print(f"Rep count: {counter}")
        except Exception as e:
            print(f"Error processing landmarks: {e}")

        cv2.rectangle(image, (0,0), (640,40), ORANGE, -1)
        
        status_text = [
            ("REPS", str(counter), (15,25), (90,25)),
            ("STAGE", stage if stage else "None", (160,25), (235,25))
        ]
        
        for label, value, label_pos, value_pos in status_text:
            cv2.putText(image, label, label_pos, FONT, 0.5, BLACK, 1, cv2.LINE_AA)
            cv2.putText(image, value, value_pos, FONT, 1, WHITE, 2, cv2.LINE_AA)

        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=RED, thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=BLACK, thickness=2, circle_radius=2)
        )

        cv2.imshow('Exercise Counter', image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
    
