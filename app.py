from flask import Flask, render_template, request, send_from_directory
from PIL import Image
import torch
from torchvision import transforms, models
import torch.nn as nn
import os

app = Flask(__name__)

# 모델 예측 함수 정의
def predict_image(image_path):
    transforms_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 모델 load
    loaded_model = models.resnet34(pretrained=False)
    num_features = loaded_model.fc.in_features
    loaded_model.fc = nn.Linear(num_features, 10)
    
    # 모델을 CPU에 매핑하여 로드
    device = torch.device('cpu')
    loaded_model.load_state_dict(torch.load('trained_model.2.pth', map_location=device))
    loaded_model.eval()

    # 이미지 열기 및, transforms 함수를 이용한 전처리
    image = Image.open(image_path)
    image_tensor = transforms_test(image).unsqueeze(0)

    # 모델 예측
    with torch.no_grad():
        outputs = loaded_model(image_tensor)
        _, preds = torch.max(outputs, 1)
        confidence = torch.nn.functional.softmax(outputs, dim=1)[0][preds].item()

    # 결과 반환
    return preds.item(), confidence

# 이미지 업로드 및 예측을 처리하는 메인 라우트
@app.route('/', methods=['GET', 'POST'])
def index():

    # 이미지 파일이 POST 요청으로 제출되었는지 확인
    if request.method == 'POST':
        file = request.files['file']

        if file:
            # 디렉토리 생성 및 업로드된 파일 저장
            if not os.path.exists('static/uploads'):
                os.makedirs('static/uploads')

            file_path = 'static/uploads/uploaded_image.jpg'
            file.save(file_path)

            # 정의된 모델 예측 함수를 이용해 결과 값 반환
            prediction, confidence = predict_image(file_path)

            confidence_threshold = 0.5

            # 예측 신뢰도에 따라 결과를 표시
            if confidence >= confidence_threshold:
                class_names = ['Bunsen Burner(분젠 버너)', 'Vernier Calipers(버니어 캘리퍼스)', 'centrifuge(원심분리기)', 'conway diffusion cell(콘웨이 디쉬)', 'desiccator(데시케이터)', 'evaporating dish(증발접시)', 'lab viscometer(점도계)', 'syringe filter(시린지 필터)', 'vortex mixer(볼텍스 믹서)', 'watch glass(시계접시)']
                result = class_names[prediction]
                return render_template('index.html', result=result, image_path=file_path)

    # 이미지가 업로드되지 않았을 때 메인 페이지 렌더링
    return render_template('index.html', result=None, image_path=None)

# Flask 앱 실행
if __name__ == '__main__':
    app.run(debug=True)