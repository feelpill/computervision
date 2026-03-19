# Computer Vision Week 3 Assignment

OpenCV와 Python을 이용한 **컴퓨터 비전 기초 과제**입니다.  
이번 실습에서는 **Sobel Edge Detection, Hough Line Transform, GrabCut Segmentation**을 구현했습니다.

---

# Development Environment

* Python 3.x
* OpenCV (cv2)
* NumPy
* OS : Windows 10

---

# Project Structure

```text
computervision
│
├── 1week
│
├── 2week
│
├── 3week
│   ├── ex1_sobeledge.py
│   ├── ex2_houghlines.py
│   ├── ex3_grabcut.py
│   │
│   ├── edgeDetectionimage.jpg
│   ├── dabo.jpg
│   ├── coffee cup.jpg
│   │
│   ├── result
│   │   ├── ex1_result.jpg
│   │   ├── ex2_result.jpg
│   │   └── ex3_result.jpg
│   │
│   └── README.md
│
└── README.md
```
# Assignment 1 : Sobel Edge Detection
Description
소벨 필터(Sobel Filter)를 이용하여 이미지의 에지(Edge) 를 검출한다.
이미지의 밝기가 급격하게 변하는 부분을 찾기 위해 x축과 y축 방향의 미분을 계산하고, 이를 기하학적으로 합성하여 최종적인 에지 강도(Magnitude) 이미지를 생성한다.

# Code Explanation
1. x, y 방향 에지 검출 및 강도 계산 (Core Algorithm)
```
# 1. x, y 방향 미분 (음수 미분값 보존을 위해 CV_64F 사용)
sobel_x = cv.Sobel(grayImage, cv.CV_64F, 1, 0, ksize=3)
sobel_y = cv.Sobel(grayImage, cv.CV_64F, 0, 1, ksize=3)

# 2. 미분 벡터 크기 합성 및 화면 출력을 위한 uint8 포맷 변환
gradient_magnitude = cv.magnitude(sobel_x, sobel_y)
sobel_combined = cv.convertScaleAbs(gradient_magnitude)
```

2. 결과 이미지 로드 및 팝업창 출력
```
# 저장된 결과 이미지(ex1_result.jpg)를 불러와서 화면에 띄우기
result_img = cv.imread('result/ex1_result.jpg')

if result_img is not None:
    cv.imshow('Sobel Edge Result', result_img)
    cv.waitKey(0)
    cv.destroyAllWindows()
else:
    print("결과 이미지를 찾을 수 없습니다.")
 ```
# Assignment 2 : Hough Line Transform

Description
Canny 알고리즘과 확률적 허프 변환(Probabilistic Hough Transform) 을 이용하여 이미지 내의 직선(선분) 을 검출한다. 에지 픽셀들을 파라미터 공간으로 변환하여, 일직선상에 픽셀들이 겹치는 지점을 기하학적 선분으로 판별한다

# Code Explanation
1. Canny 에지 추출 및 확률적 허프 변환 (Core Algorithm)
```
# 1. 선명한 윤곽선(Edge Map) 추출
edges = cv.Canny(gray_img, 100, 200)

# 2. 허프 변환으로 선분 검출 (최소 길이 50, 끊어진 간격 최대 10픽셀 허용)
lines = cv.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=50, minLineLength=50, maxLineGap=10)

# 3. 원본 이미지에 붉은색(0, 0, 255)으로 직선 그리기
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv.line(line_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
```
2. 결과 이미지 로드 및 팝업창 출력
```
# 저장된 결과 이미지(ex2_result.jpg)를 불러와서 화면에 띄우기
result_img2 = cv.imread('result/ex2_result.jpg')

if result_img2 is not None:
    cv.imshow('Hough Line Transform Result', result_img2)
    cv.waitKey(0)
    cv.destroyAllWindows()
```


# Assignment 3 : GrabCut Segmentation
Description
GrabCut 알고리즘을 이용하여 대화식으로 전경(객체)과 배경을 분리한다.
사용자가 객체를 감싸는 사각형(Bounding Box)을 지정하면, 사각형 외부를 '확실한 배경'으로 가정하고 통계적 모델을 학습하여 객체만 추출해 낸다.

# Code Explanation
1. GrabCut 알고리즘 및 마스크 이진화 (Core Algorithm)
```
# 1. 사각형 기준으로 GrabCut 초기화 및 모델 학습 (5회 반복)
cv.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv.GC_INIT_WITH_RECT)

# 2. 결과 마스크에서 배경(0, 2)은 0으로, 전경(1, 3)은 1로 치환하여 이진화
mask2 = np.where((mask == cv.GC_BGD) | (mask == cv.GC_PR_BGD), 0, 1).astype('uint8')

# 3. 마스크 차원(3D) 확장 후 원본 이미지에 곱하여 배경 제거
extracted_img = img * mask2[:, :, np.newaxis]
```

2. 결과 이미지 로드 및 팝업창 출력
```
# 저장된 결과 이미지(ex3_result.jpg)를 불러와서 화면에 띄우기
result_img3 = cv.imread('result/ex3_result.jpg')

if result_img3 is not None:
    cv.imshow('GrabCut Result', result_img3)
    cv.waitKey(0)
    cv.destroyAllWindows()
```

# How to Run
```
python ex1_sobeledge.py
python ex2_houghlines.py
python ex3_grabcut.py
```
