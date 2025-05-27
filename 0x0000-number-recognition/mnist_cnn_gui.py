import numpy as np
import tensorflow as tf
import cv2 as cv
import matplotlib.pyplot as plt

# 모델 로드
model = tf.keras.models.load_model('mnist_cnn_model.h5')

# 초기화 함수
def reset():
    global img
    img = np.ones((200, 520, 3), dtype=np.uint8) * 255
    cv.putText(img, 'e:erase, s:show, r:recognition, q:quit',
               (10, 40), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1)
    cv.rectangle(img, (10, 50), (110, 150), (0, 0, 0), 1)

# 숫자 추출 함수
def grab_numeral():
    roi = img[51:149, 11:109, 0]  # 단일 숫자 영역
    roi = 255 - cv.resize(roi, (28, 28), interpolation=cv.INTER_CUBIC)
    return roi

# 숫자 확인용 시각화 함수
def show():
    numeral = grab_numeral()
    plt.figure(figsize=(5, 5))
    plt.imshow(numeral, cmap='gray')
    plt.xticks([])
    plt.yticks([])
    plt.show()

# 숫자 인식 함수
def recognition():
    numeral = grab_numeral()
    numeral = numeral.reshape(1, 28, 28, 1).astype(np.float32) / 255.0
    res = model.predict(numeral, verbose=0)
    class_id = np.argmax(res, axis=1)[0]
    cv.putText(img, f'Predict: {class_id}', (120, 100), cv.FONT_HERSHEY_SIMPLEX,
               1.2, (255, 0, 0), 2)

# 마우스로 쓰기 위한 콜백 함수
def writing(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN or (event == cv.EVENT_MOUSEMOVE and flags == cv.EVENT_FLAG_LBUTTON):
        cv.circle(img, (x, y), 3, (0, 0, 0), -1)

# 초기화 및 윈도우 설정
reset()
cv.namedWindow('숫자 인식')
cv.setMouseCallback('숫자 인식', writing)

# 메인 루프
while True:
    cv.imshow('숫자 인식', img)
    key = cv.waitKey(1)
    if key == ord('e'):
        reset()
    elif key == ord('s'):
        show()
    elif key == ord('r'):
        recognition()
    elif key == ord('q'):
        break

cv.destroyAllWindows()
