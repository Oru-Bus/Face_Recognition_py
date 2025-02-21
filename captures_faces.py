import cv2
import mediapipe as mp
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder


mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.8)


DATASET_PATH = "datasets"
os.makedirs(DATASET_PATH, exist_ok=True)

name = input("UserName : ").strip()
user_path = os.path.join(DATASET_PATH, name)
os.makedirs(user_path, exist_ok=True)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
count = 0

while count < 500:
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb_frame)

    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = frame.shape
            x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)

            face = frame[y:y+h, x:x+w]
            if face.size > 0:
                face_resized = cv2.resize(face, (160, 160))
                cv2.imwrite(f"{user_path}/{count}.jpg", face_resized)
                count += 1

    cv2.imshow("Recording faces", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Recording complete.")
