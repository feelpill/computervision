import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# 1. cv.imread()를 사용하여 이미지를 불러옴
# 절대 경로를 사용할 때는 경로 앞에 'r'을 붙여 역슬래시(\) 오류를 방지합니다.
image_path = r"C:\computervision\3week\edgeDetectionimage.jpg"
edgeDetectionImage = cv.imread(image_path)

# (방어 코드) 이미지가 제대로 불러와지지 않았을 때 원인을 알려줍니다.
if edgeDetectionImage is None:
    print(f"오류: '{image_path}' 경로에서 이미지를 찾을 수 없습니다.")
    print("해당 폴더에 파일이 있는지, 파일명이나 확장자(.jpg, .jpeg 등)가 정확한지 확인해 주세요.")
    exit()

# 2. cv.cvtColor()를 사용하여 그레이스케일로 변환
grayImage = cv.cvtColor(edgeDetectionImage, cv.COLOR_BGR2GRAY)

# 3. cv.Sobel()을 사용하여 x축과 y축 방향의 에지를 검출
# 요구사항: x축(cv.CV_64F, 1, 0), y축(cv.CV_64F, 0, 1) / 힌트: ksize=3
sobel_x = cv.Sobel(grayImage, cv.CV_64F, 1, 0, ksize=3)
sobel_y = cv.Sobel(grayImage, cv.CV_64F, 0, 1, ksize=3)

# 4. cv.magnitude()를 사용하여 에지 강도 계산
gradient_magnitude = cv.magnitude(sobel_x, sobel_y)

# 힌트: cv.convertScaleAbs()를 사용하여 에지 강도 이미지를 uint8로 변환
sobel_combined = cv.convertScaleAbs(gradient_magnitude)

# 5. Matplotlib를 사용하여 원본 이미지와 에지 강도 이미지를 나란히 시각화
# 원본 이미지를 Matplotlib에서 올바른 색상으로 보기 위해 BGR -> RGB 변환
original_rgb = cv.cvtColor(edgeDetectionImage, cv.COLOR_BGR2RGB)

plt.figure(figsize=(10, 5))

# 원본 이미지 시각화
plt.subplot(1, 2, 1)
plt.imshow(original_rgb)
plt.title('Original Image')
plt.axis('off') # 눈금자 숨기기

# 에지 강도 이미지 시각화 (힌트: cmap='gray' 사용)
plt.subplot(1, 2, 2)
plt.imshow(sobel_combined, cmap='gray')
plt.title('Sobel Edge Magnitude')
plt.axis('off')

# 화면에 출력
plt.tight_layout()
plt.show()