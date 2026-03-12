# Computer Vision Week 2 Assignment

OpenCV와 Python을 이용한 **중급 컴퓨터 비전 과제**입니다.  
이번 실습에서는 **Camera Calibration, Affine Transformation, Stereo Depth Estimation**을 구현했습니다.

---

# Development Environment

* Python 3.11.8
* OpenCV (cv2)
* NumPy
* OS : Windows 10

---

# Project Structure

```
computervision
│
├── week1
│
├── week2
│   ├── ex1_calibrate.py
│   ├── ex2_rotatetransform.py
│   ├── ex3_disparity.py
│   │
│   ├── calibration_images
│   │
│   ├── rose.png
│   ├── rotatetransformrose.png
│   ├── left.png
│   ├── right.png
│   ├── undistorted_result.png
│   ├── ex3_totalresult.png
│   │
│   ├── outputs
│   │   ├── disparity_color.png
│   │   ├── depth_color.png
│   │   ├── left_roi.png
│   │   └── right_roi.png
│   │
│   └── README.md
│
└── README.md
```

---

# Assignment 1 : Camera Calibration

## Description

체커보드 이미지를 이용하여 **카메라 캘리브레이션(Camera Calibration)** 을 수행한다.

카메라 캘리브레이션은 다음 정보를 계산하기 위해 사용된다.

* Camera Matrix (카메라 내부 파라미터)
* Distortion Coefficients (렌즈 왜곡 계수)
* Reprojection Error (캘리브레이션 정확도)

이를 통해 **렌즈 왜곡이 제거된 이미지**를 생성할 수 있다.

---

## Key Function

```
cv.findChessboardCorners()
cv.cornerSubPix()
cv.calibrateCamera()
cv.projectPoints()
cv.undistort()
```

---

## Code Explanation

### 1. 체커보드 설정

```
CHECKERBOARD = (9,6)
square_size = 25
```

| 변수 | 설명 |
|-----|------|
| CHECKERBOARD | 체커보드 내부 코너 개수 |
| square_size | 체커보드 한 칸의 실제 크기 |

---

### 2. 체커보드 코너 검출

이미지에서 체커보드 코너를 검출한다.

```
cv.findChessboardCorners()
```

코너 검출이 성공하면 실제 좌표와 이미지 좌표를 저장한다.

---

### 3. 코너 정밀화

```
cv.cornerSubPix()
```

코너 위치를 **subpixel 수준으로 보정**하여 캘리브레이션 정확도를 높인다.

---

### 4. 카메라 캘리브레이션

```
cv.calibrateCamera()
```

다음 값을 계산한다.

* Camera Matrix
* Distortion Coefficients
* Rotation Vector
* Translation Vector

---

### 5. Reprojection Error 계산

```
cv.projectPoints()
```

실제 코너 위치와 다시 투영된 코너 위치의 차이를 계산한다.

Error 값이 **작을수록 정확한 캘리브레이션**이다.

---

## Result

### Checkerboard Detection

![corners](corners_result.jpg)

---

### Original Image

![original](calibration_images/left.jpg)

---

### Undistorted Image

![undistorted](undistorted_result.jpg)

---

# Assignment 2 : Affine Transformation

## Description

이미지에 **Affine Transform (아핀 변환)** 을 적용하여  
이미지를 **회전, 스케일, 평행이동** 하는 프로그램을 구현한다.

Affine Transform은 다음 변환을 포함한다.

* Rotation (회전)
* Scaling (크기 변화)
* Translation (평행 이동)

---

## Key Function

```
cv.getRotationMatrix2D()
cv.warpAffine()
```

---

## Code Explanation

### 1. 이미지 불러오기

```
img = cv2.imread("rose.png")
```

원본 이미지를 읽어온다.

---

### 2. 이미지 중심 계산

```
center = (w//2, h//2)
```

이미지 중심을 회전 기준점으로 설정한다.

---

### 3. 회전 + 스케일 변환 행렬 생성

```
M = cv2.getRotationMatrix2D(center, 30, 0.8)
```

| 값 | 의미 |
|---|---|
| 30 | 30도 회전 |
| 0.8 | 80% 크기로 축소 |

---

### 4. 평행 이동 추가

```
M[0,2] += 80
M[1,2] += -40
```

| 이동 | 의미 |
|---|---|
| +80 | 오른쪽 이동 |
| -40 | 위쪽 이동 |

---

### 5. Affine 변환 적용

```
cv2.warpAffine()
```

이미지에 회전 + 스케일 + 이동 변환을 적용한다.

---

## Result

Original Image

![rose]rose.png)

---

Transformed Image

![affine](rotatetransformrose.png)

---

# Assignment 3 : Stereo Depth Estimation

## Description

Stereo Vision을 이용하여 **Depth Map (거리 정보)** 를 계산한다.

좌우 카메라 이미지의 **disparity(시차)** 를 계산하고  
다음 공식을 이용하여 **depth(거리)** 를 계산한다.

```
Z = fB / d
```

| 변수 | 의미 |
|---|---|
| f | focal length |
| B | baseline |
| d | disparity |

---

## Key Function

```
cv.StereoBM_create()
cv.applyColorMap()
cv.rectangle()
cv.putText()
```

---

## Code Explanation

### 1. Stereo Matching

```
stereo = cv2.StereoBM_create()
```

StereoBM 알고리즘을 사용하여 **disparity map**을 계산한다.

---

### 2. Depth 계산

```
depth = fB / disparity
```

disparity 값이 클수록 **물체는 카메라에 가까움**을 의미한다.

---

### 3. ROI 설정

특정 물체 영역을 지정하여 평균 depth를 계산한다.

```
Painting
Frog
Teddy
```

각 ROI 영역에서 **평균 disparity / depth** 값을 계산한다.

---

### 4. Disparity 시각화

```
cv.applyColorMap()
```

색상으로 깊이 정보를 표현한다.

* 빨강 → 가까움
* 파랑 → 멀음

---

## Result

### Disparity Map

![disparity](outputs/disparity_color.png)

---

### Depth Map

![depth](outputs/depth_color.png)

---

### ROI Detection

![roi](outputs/left_roi.png)
![roi](outputs/right_roi.png)

---

# How to Run

```
python ex1_calibrate.py
python ex2_rotatestransform.py
python ex3_disparity.py
```

---

# Learning Outcome

이번 과제를 통해 다음 내용을 학습하였다.

* Camera Calibration
* Affine Transformation
* Stereo Vision
* Disparity & Depth Map
* OpenCV 기반 3D 거리 계산

---

# Author

Computer Vision Assignment  
Dong-A University
