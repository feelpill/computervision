# Computer Vision Week 2 Assignment

OpenCV와 Python을 이용한 **중급 컴퓨터 비전 과제**입니다.  
이번 과제에서는 **카메라 캘리브레이션, 이미지 기하 변환, 스테레오 깊이 추정**을 구현했습니다.

---

# Development Environment

* Python 3.11.8
* OpenCV (cv2)
* NumPy
* OS : Windows 10

---

# Project Structure


computervision
│
├── week1
│ └── (Week1 과제 코드)
│
├── week2
│ ├── ex1_calibrate.py
│ ├── ex2_affine_transform.py
│ ├── ex3_stereo_depth.py
│ ├── calibration_images
│ ├── left.png
│ ├── right.png
│ └── rose.png
│
└── README.md


---

# Assignment 1 : Camera Calibration

## Description

체커보드 패턴 이미지를 이용하여 **카메라 내부 파라미터(Camera Matrix)**와  
**렌즈 왜곡 계수(Distortion Coefficients)**를 계산하는 프로그램을 구현하였다.

카메라 캘리브레이션은 실제 3D 좌표와 이미지 2D 좌표의 관계를 이용하여  
카메라의 기하학적 특성을 추정하는 과정이다.

---

## Key Function


cv2.findChessboardCorners()
cv2.cornerSubPix()
cv2.calibrateCamera()
cv2.projectPoints()
cv2.undistort()


---

## Code Explanation

### 1. 체커보드 코너 검출


cv2.findChessboardCorners()


체커보드 패턴에서 내부 코너 위치를 탐지한다.

검출된 코너는 이미지 좌표(`imgpoints`)로 저장된다.

---

### 2. 코너 정밀화


cv2.cornerSubPix()


검출된 코너 위치를 **subpixel 수준으로 보정**하여 정확도를 향상시킨다.

---

### 3. 카메라 파라미터 계산


cv2.calibrateCamera()


다음 파라미터를 계산한다.

* Camera Matrix (K)
* Distortion Coefficients
* Rotation Vector
* Translation Vector

---

### 4. Reprojection Error 계산

캘리브레이션 정확도를 평가하기 위해 **Reprojection Error**를 계산한다.


cv2.projectPoints()


3D 좌표를 다시 이미지 평면에 투영하여  
실제 코너 위치와의 차이를 측정한다.

---

### 5. 이미지 왜곡 보정


cv2.undistort()


계산된 카메라 파라미터를 이용하여  
렌즈 왜곡이 보정된 이미지를 생성한다.

---

## Result

* Camera Matrix 출력
* Distortion Coefficient 출력
* Reprojection Error 계산
* 왜곡 보정 이미지 생성

---

# Assignment 2 : Affine Transformation

## Description

이미지에 **회전(Rotation), 크기 조절(Scaling), 평행 이동(Translation)**을 동시에 적용하는 프로그램을 구현하였다.

---

## Transformation Requirement

* 이미지 중심 기준 **30도 회전**
* **0.8배 크기 축소**
* **x 방향 +80px 이동**
* **y 방향 -40px 이동**

---

## Key Function


cv2.getRotationMatrix2D()
cv2.warpAffine()


---

## Code Explanation

### 1. 회전 + 스케일 변환 행렬 생성


cv2.getRotationMatrix2D(center, 30, 0.8)


이미지 중심을 기준으로

* 30도 회전
* 0.8배 스케일

Affine 변환 행렬을 생성한다.

---

### 2. 평행 이동 추가


M[0,2] += 80
M[1,2] += -40


Affine 행렬의 translation 값을 변경하여  
이미지를 이동시킨다.

---

### 3. Affine 변환 적용


cv2.warpAffine()


변환 행렬을 이용하여  
회전, 크기 조절, 평행 이동을 동시에 수행한다.

---

## Result

* 이미지 회전
* 이미지 축소
* 위치 이동 적용된 결과 이미지 출력

---

# Assignment 3 : Stereo Depth Estimation

## Description

좌우 카메라 이미지의 **Disparity**를 이용하여  
물체의 **Depth(거리)**를 계산하는 프로그램을 구현하였다.

---

## Depth Formula


Z = fB / d


Where

* **Z** : Depth (거리)
* **f** : Camera focal length
* **B** : Baseline (카메라 간 거리)
* **d** : Disparity

---

## Key Function


cv2.StereoBM_create()
cv2.applyColorMap()
cv2.rectangle()


---

## Code Explanation

### 1. Disparity 계산


cv2.StereoBM_create()


좌우 이미지의 블록 매칭을 통해  
픽셀 간 위치 차이(disparity)를 계산한다.

---

### 2. Depth Map 계산

Depth 공식


Z = fB / d


을 이용하여 각 픽셀의 거리를 계산한다.

---

### 3. ROI 거리 계산

이미지의 특정 영역(ROI)을 지정하여  
각 객체의 평균 disparity와 depth를 계산한다.

ROI 대상

* Painting
* Frog
* Teddy

---

### 4. Disparity / Depth 시각화


cv2.applyColorMap()


컬러맵을 이용하여

* **가까운 물체 → 빨강**
* **먼 물체 → 파랑**

으로 시각화한다.

---

## Result

* Disparity Map 생성
* Depth Map 생성
* ROI별 평균 거리 계산
* 가장 가까운 객체 / 가장 먼 객체 출력

---

# How to Run


python ex1_calibrate.py
python ex2_affine_transform.py
python ex3_stereo_depth.py


---

# Learning Outcome

이번 과제를 통해 다음 내용을 학습하였다.

* Camera Calibration
* Affine Transformation
* Stereo Vision
* Depth Estimation
* OpenCV 기반 3D 거리 계산

---

# Author

Computer Vision Assignment  
Dong-A University