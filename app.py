# 필요한 라이브러리 임포트
import os
from flask import Flask, request, render_template
import cv2
import numpy as np

app = Flask(__name__)

# YOLO 모델 및 가중치 파일 경로 설정
yolo_cfg = 'yolov4.cfg'  # YOLO 모델 설정 파일 경로
yolo_weights = 'yolov4.weights'  # YOLO 가중치 파일 경로
yolo_classes = 'coco.names'  # 클래스 이름 파일 경로

# YOLO 모델 초기화
net = cv2.dnn.readNet(yolo_weights, yolo_cfg)
with open(yolo_classes, 'r') as f:
    classes = f.read().strip().split('\n')

# 이미지를 업로드하고 감지하는 루트 경로
@app.route('/', methods=['GET', 'POST'])
def index():
    result = None

    if request.method == 'POST':
        # 업로드된 이미지를 가져옴
        uploaded_image = request.files['image']
        if uploaded_image:
            image_path = 'static/uploaded_image.jpg'  # 이미지를 저장할 경로
            uploaded_image.save(image_path)

            # 이미지를 YOLO 모델로 로드하여 감지
            image = cv2.imread(image_path)
            height, width = image.shape[:2]
            blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
            net.setInput(blob)
            layer_names = net.getUnconnectedOutLayersNames()
            detections = net.forward(layer_names)

            # 중복 감지를 방지하기 위한 변수 및 리스트 초기화
            max_confidence = 0.01  # 매우 낮은 확률 임계값 설정
            detected_objects = []

            # 감지된 객체를 이미지에 표시
            for detection in detections:
                for obj in detection:
                    scores = obj[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > max_confidence:
                        center_x = int(obj[0] * width)
                        center_y = int(obj[1] * height)
                        w = int(obj[2] * width)
                        h = int(obj[3] * height)
                        x = center_x - w // 2
                        y = center_y - h // 2

                        # 중복된 객체를 방지하기 위해 기존에 감지된 객체와 겹치지 않는지 확인
                        is_overlapping = False
                        for (x1, y1, w1, h1, class_id1) in detected_objects:
                            if (x < x1 + w1 and x + w > x1 and y < y1 + h1 and y + h > y1):
                                is_overlapping = True
                                break
                        if not is_overlapping:
                            detected_objects.append((x, y, w, h, class_id))
                            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                            label = f'{classes[class_id]}: {confidence:.2f}'
                            cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            result = 'static/detected_image.jpg'
            cv2.imwrite(result, image)

    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
