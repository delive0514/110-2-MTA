import numpy as np
import cv2
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

label = np.uint8(np.loadtxt("label.txt"))
cap = cv2.VideoCapture("test_dataset.avi")
success= True
Video=[]
while(success):
    
    success, frame = cap.read()

    if not success:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    Video.append(gray)
    

data = np.array(Video).reshape((len(Video),-1))

clf = KNeighborsClassifier()

train_x, test_x, train_y, test_y = train_test_split(data, label, test_size=0.5)

clf.fit(train_x, train_y)

predicted = clf.predict(test_x)


print(
    f"Classification report for classifier {clf}:\n"
    f"{metrics.classification_report(test_y, predicted)}\n"
)