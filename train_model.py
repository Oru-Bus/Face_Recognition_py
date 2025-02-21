import os
import cv2
import pickle
import numpy as np
from sklearn.svm import SVC
from keras_facenet import FaceNet


embedder = FaceNet()


dataset_path = "datasets"


embeddings = []
labels = []


for person_name in os.listdir(dataset_path):
    person_path = os.path.join(dataset_path, person_name)
    if not os.path.isdir(person_path):
        continue


    for image_name in os.listdir(person_path):
        image_path = os.path.join(person_path, image_name)
        image = cv2.imread(image_path)
        if image is None:
            continue


        faces = embedder.embeddings([image])
        if len(faces) > 0:
            embeddings.append(faces[0])
            labels.append(person_name)


embeddings = np.array(embeddings)
labels = np.array(labels)


classifier = SVC(kernel='linear', probability=True)
classifier.fit(embeddings, labels)


with open("face_recognizer.pkl", "wb") as f:
    pickle.dump(classifier, f)

print("Training completed.")