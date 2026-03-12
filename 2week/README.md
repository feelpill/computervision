# Computer Vision Week 2 Assignment

OpenCV와 Python을 이용한 **카메라 캘리브레이션(Camera Calibration)** 실습입니다.  
체커보드 이미지를 이용하여 **카메라 내부 파라미터(Camera Matrix)** 와 **렌즈 왜곡 계수(Distortion Coefficients)** 를 계산하고  
이미지의 **렌즈 왜곡을 보정(Undistortion)** 하는 프로그램을 구현했습니다.

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
│   ├── ex1_calibration.py
│   ├── calibration_images
│   │   ├── left01.jpg
│   │   ├── left02.jpg
│   │   └── ...
│   │
│   ├── undistorted_result.jpg
│   └── README.md
│
└── README.md
```

---

# Assignment : Camera Calibration

## Description

체커보드 이미지를 이용하여 **카메라 캘리브레이션(Camera Calibration)** 을 수행한다.

카메라 캘리브레이션은 다음 정보를 계산하기 위해 사용된다.

* Camera Matrix (카메라 내부 파라미터)
* Distortion Coefficients (렌즈 왜곡 계수)
* Reprojection Error (캘리브레이션 정확도)

이를 통해 **렌즈 왜곡을 제거한 이미지**를 생성할 수 있다.

---

# Key Function

```
cv.findChessboardCorners()
cv.cornerSubPix()
cv.calibrateCamera()
cv.projectPoints()
cv.undistort()
```

---

# Code Explanation

## 1. 체커보드 설정

체커보드 내부 코너 개수를 설정한다.

```
CHECKERBOARD = (9,6)
square_size = 25
```

| 변수 | 설명 |
|-----|------|
| CHECKERBOARD | 체커보드 내부 코너 개수 |
| square_size | 체커보드 한 칸의 실제 크기 (mm) |

---

## 2. 체커보드 3D 좌표 생성

실제 세계 좌표를 생성한다.

```
objp = np.zeros((CHECKERBOARD[0]*CHECKERBOARD[1],3), np.float32)
objp[:,:2] = np.mgrid[0:CHECKERBOARD[0],0:CHECKERBOARD[1]].T.reshape(-1,2)
objp *= square_size
```

체커보드는 **Z = 0 평면 위에 있다고 가정**한다.

---

## 3. 체커보드 코너 검출

이미지에서 체커보드 코너를 검출한다.

```
ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)
```

코너 검출이 성공하면

* 실제 좌표 저장
* 이미지 좌표 저장

---

## 4. 코너 정밀화

코너 위치를 **Subpixel 수준으로 보정**한다.

```
corners2 = cv2.cornerSubPix()
```

이를 통해 **캘리브레이션 정확도를 높인다.**

---

## 5. 카메라 캘리브레이션

다음 함수로 카메라 파라미터를 계산한다.

```
cv.calibrateCamera()
```

출력 결과

* Camera Matrix
* Distortion Coefficients
* Rotation Vector
* Translation Vector

---

## 6. Reprojection Error 계산

캘리브레이션 정확도를 평가하기 위해  
**Reprojection Error** 를 계산한다.

```
imgpoints2, _ = cv2.projectPoints()
```

실제 코너와 다시 투영된 코너의 차이를 계산한다.

값이 **작을수록 캘리브레이션 정확도가 높다.**

---

## 7. 이미지 왜곡 보정

렌즈 왜곡을 제거하기 위해 다음 함수를 사용한다.

```
cv.undistort()
```

이 과정으로 **직선 왜곡이 제거된 이미지**를 얻을 수 있다.

---

# Result

## Checkerboard Corner Detection

체커보드 코너 검출 결과

![Corners](week2/corners.png)

---

## Original Image

![Original](week2/original.jpg)

---

## Undistorted Image

렌즈 왜곡이 제거된 결과 이미지

![Undistorted](week2/undistorted_result.jpg)

---

# How to Run

```
python ex1_calibration.py
```

---

# Learning Outcome

이번 실습을 통해 다음 내용을 학습하였다.

* 카메라 캘리브레이션 원리
* 체커보드 코너 검출
* Camera Matrix 계산
* 렌즈 왜곡 보정
* Reprojection Error 평가

---

# Author

Computer Vision Assignment  
Dong-A University
