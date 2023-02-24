# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
import torch
import torchvision
import cv2
import time
import math
import mediapipe as mp
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torch import nn
from sklearn.metrics import classification_report

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

df = pd.read_csv('./kaggle/input/exercise-recognition/train.csv')

df['pose'].value_counts()

encoder = LabelEncoder()
y = df['pose']
y = encoder.fit_transform(y)

class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y), y=y)

scaler = MinMaxScaler()

X = df.drop(['pose_id', 'pose'], axis='columns')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, shuffle=True, random_state=2022)

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

class ExerciseDataset(Dataset):
    def __init__(self, X, y):
        self.x = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.int64))
        self.n_samples = X.shape[0]
    
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    
    def __len__(self):
        return self.n_samples
    
train_dataset = ExerciseDataset(X_train, y_train)

batch_size = 50

train_loader = DataLoader(train_dataset, batch_size=batch_size)

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        return out
    
epochs = 40
learning_rate = 0.01
hidden_size = 200

model = NeuralNet(X_train.shape[1], hidden_size, len(class_weights))

criterion = nn.CrossEntropyLoss(weight=torch.from_numpy(class_weights.astype(np.float32)))
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

n_total_steps = len(train_loader)

for epoch in range(epochs):
    for i, (features, labels) in enumerate(train_loader):
        outputs = model(features)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % 6 == 0:
            print(f'epoch {epoch+1} / {epochs}, step {i+1} / {n_total_steps}, loss = {loss.item():.4f}')

test_features = torch.from_numpy(X_test.astype(np.float32))
test_labels = y_test
with torch.no_grad():
    outputs = model(test_features)
    _, predictions = torch.max(outputs, 1)

print(classification_report(test_labels, predictions, target_names=encoder.classes_))

class poseDetector():
    def __init__(self, mode = False, modelCom = 1, upBody = False, smooth = True, detectionCon = 0.5, trackCon = 0.5):
        self.mode = mode
        self.modelCom = modelCom
        self.upBody = upBody
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.mode, self.modelCom, self.upBody, self.smooth, self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def findPose(self, img, draw = True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)

        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
        return img

    def findPosition(self, img, draw = True):
        self.lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 7, (0, 255, 0), cv2.FILLED)
                    if id == 23:
                        cv2.circle(img, (cx, cy), 10, (255, 0, 0), cv2.FILLED)
                    if id == 24:
                        cv2.circle(img, (cx, cy), 10, (255, 0, 0), cv2.FILLED)
                    if id == 25:
                        cv2.circle(img, (cx, cy), 10, (255, 0, 0), cv2.FILLED)
                    if id == 26:
                        cv2.circle(img, (cx, cy), 10, (255, 0, 0), cv2.FILLED)
        return self.lmList

    def findAngle(self, img, p1, p2, p3, draw = True):
        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        x3, y3 = self.lmList[p3][1:]

        angle = math.degrees(math.atan2(y3-y2, x3-x2) - math.atan2(y1-y2, x1-x2))
        if angle < 0:
            angle += 360

        # if draw:
        #     cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 5)
        #     cv2.line(img, (x2, y2), (x3, y3), (255, 255, 255), 5)
        #     cv2.circle(img, (x1, y1), 10, (0, 255, 0), cv2.FILLED)
        #     cv2.circle(img, (x1, y1), 15, (0, 0, 255), 2)
        #     cv2.circle(img, (x2, y2), 10, (0, 255, 0), cv2.FILLED)
        #     cv2.circle(img, (x2, y2), 15, (0, 0, 255), 2)
        #     cv2.circle(img, (x3, y3), 10, (0, 255, 0), cv2.FILLED)
        #     cv2.circle(img, (x3, y3), 15, (0, 0, 255), 2)
        #     cv2.putText(img, str(int(angle)), (x2 - 20, y2 + 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255,), 2)

        return angle


cap = cv2.VideoCapture(0)
pTime = 0
detector = poseDetector()

while True:
    success, img = cap.read()

    img = detector.findPose(img)
    lmList = detector.findPosition(img)
    coords = []
    if detector.results.pose_landmarks:
        for lm in detector.results.pose_landmarks.landmark:
                    coords.append(lm.x)
                    coords.append(lm.y)
                    coords.append(lm.z)
        # coords = [[lmk.x, lmk.y, lmk.z] for lmk in detector.results.pose_landmarks.landmark]
        print(coords)
        with torch.no_grad():
            outputs = model(torch.Tensor([coords]))
            print(outputs)
            _, prediction = torch.max(outputs, 1)
        print(prediction)
        print(encoder.classes_[prediction])
        cv2.putText(img, str(encoder.classes_[prediction]), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3)
    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime

    

    # cv2.putText(img, "FPS: " + str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3)

    cv2.imshow("Image", img)
    key = cv2.waitKey(1)

    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
