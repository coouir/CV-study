import cv2 as cv
import numpy as np
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions

# 모델 불러오기
model = ResNet50(weights='imagenet')

img = cv.imread('flamingo.jpg')
img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)

# 리사이즈 및 전처리
x = cv.resize(img_rgb, (224, 224))
x = np.reshape(x, (1, 224, 224, 3))
x = preprocess_input(x)

# 예측
preds = model.predict(x)
top5 = decode_predictions(preds, top=5)[0]
print("예측 결과:", top5)