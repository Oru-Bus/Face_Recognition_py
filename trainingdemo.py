
import cv2, os
import numpy as np
from PIL import Image


recognizer = cv2.face.LBPHFaceRecognizer.create()

id = '42'
path = f'./datasets/{id}'

def getImagesId(path):
    imagePath = [os.path.join(path, f) for f in os.listdir(path)]
    faces = []
    ids = []
    
    for imagePaths in imagePath:
        faceImage = Image.open(imagePaths).convert('L')
        faceNP = np.array(faceImage, dtype=np.uint8)
        faces.append(faceNP)
        ids.append(int(id))
        cv2.imshow("Training", faceNP)
        cv2.waitKey(1)
    return ids, faces

IDs, faceData = getImagesId(path)
recognizer.train(faceData, np.array(IDs))
recognizer.write("Trainer.yml")
cv2.destroyAllWindows()
print("Training Completed")
