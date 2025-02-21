import cv2
import numpy as np
import os
from joblib import load
from keras_facenet import FaceNet


embedder = FaceNet()

classifier = load('face_recognizer.pkl')

CONFIDENCE_THRESHOLD = 0.98

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)


while True:
    ret, frame = cap.read()
    if not ret:
        print("Échec de la capture de l'image")
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(rgb_frame, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        face = rgb_frame[y:y+h, x:x+w]
        face_resized = cv2.resize(face, (160, 160))
        face_resized = np.expand_dims(face_resized, axis=0)


        embedding = embedder.embeddings(face_resized)
        

        if hasattr(classifier, 'predict_proba'):
            proba = classifier.predict_proba(embedding)
            max_proba = np.max(proba)
            prediction = classifier.classes_[np.argmax(proba)]
        else:
            prediction = classifier.predict(embedding)
            max_proba = 1.0

        print(f"Confiance : {max_proba}")

        name = "Unknown"
        if max_proba > CONFIDENCE_THRESHOLD:
            name = prediction

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow("Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()