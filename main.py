
import cv2
import os


video = cv2.VideoCapture(0)
faceDetect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
id = input("Enter your ID : \n")
count = 0
os.makedirs(f'./datasets/{id}')

while True:
    ret, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceDetect.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:
        count += 1
        cv2.imwrite(f'./datasets/{str(id)}/{str(count)}.jpg', gray[y: y+h, x: x+w])
        cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 1)
    
    cv2.imshow("Caméra", frame)
    k = cv2.waitKey(1)
    
    if count == 500:
        break
    
video.release()
cv2.destroyAllWindows()
print("Dataset collection Done.")
