import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# 1. 이미지 불러오기 (경로 및 파일명 지정)
image_path = r"C:\computervision\3week\dabo.jpg"
original_img = cv.imread(image_path)

if original_img is None:
    print(f"오류: '{image_path}' 경로에서 이미지를 찾을 수 없습니다.")
    print("파일명이 dabo.jpg가 맞는지 확인해 주세요.")
    exit()

# 원본 이미지를 복사하여 선을 그릴 스케치북(결과용 이미지) 생성
# (원본을 보존하기 위해 copy를 사용합니다)
line_img = original_img.copy()

# 에지 검출을 위해 그레이스케일로 변환
gray_img = cv.cvtColor(original_img, cv.COLOR_BGR2GRAY)

# 2. cv.Canny()를 사용하여 에지 맵 생성
# 힌트: threshold1=100, threshold2=200
edges = cv.Canny(gray_img, 100, 200)

# 3. cv.HoughLinesP()를 사용하여 직선 검출
# 힌트: 파라미터(rho, theta, threshold, minLineLength, maxLineGap) 조정
# 아래 값들은 일반적인 기본값이므로, 다보탑 이미지에 맞게 숫자를 조절해보며 테스트하세요!
lines = cv.HoughLinesP(
    edges, 
    rho=1,                # 거리 해상도 (픽셀 단위, 보통 1)
    theta=np.pi / 180,    # 각도 해상도 (라디안 단위, 보통 1도 = np.pi/180)
    threshold=50,         # 직선으로 판단할 최소 교차점 수 (이 값이 작으면 선이 무수히 많이 검출됨)
    minLineLength=50,     # 선으로 인정할 최소 길이 (픽셀)
    maxLineGap=10         # 끊어져 있어도 하나의 선으로 이어줄 최대 픽셀 간격
)

# 4. cv.line()을 사용하여 검출된 직선을 이미지에 그림
# lines가 None이 아닐 때만(즉, 선이 1개라도 검출되었을 때만) 그리도록 방어 코드 작성
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        # 힌트: 색상은 (0, 0, 255) (BGR 기준 빨간색), 두께는 2
        cv.line(line_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
else:
    print("직선이 검출되지 않았습니다. HoughLinesP의 파라미터(threshold 등)를 낮춰보세요.")

# 5. Matplotlib를 사용하여 시각화 (나란히 배치)
# OpenCV의 BGR 색상을 Matplotlib의 RGB 색상으로 변환
original_rgb = cv.cvtColor(original_img, cv.COLOR_BGR2RGB)
line_rgb = cv.cvtColor(line_img, cv.COLOR_BGR2RGB)

plt.figure(figsize=(12, 6))

# 왼쪽: 원본 이미지
plt.subplot(1, 2, 1)
plt.imshow(original_rgb)
plt.title('Original Image')
plt.axis('off')

# 오른쪽: 허프 변환으로 직선이 그려진 이미지
plt.subplot(1, 2, 2)
plt.imshow(line_rgb)
plt.title('Detected Lines (Hough Transform)')
plt.axis('off')

plt.tight_layout()
plt.show()