# Computer Vision Week 1 Assignment

OpenCV와 Python을 이용한 **기초 컴퓨터 비전 과제**입니다.  
이미지 처리의 기본 개념인 **Grayscale 변환, 마우스 기반 Drawing, ROI (Region of Interest) 추출**을 구현했습니다.

---

# Development Environment

- Python 3.11.8
- OpenCV (cv2)
- NumPy
- OS : Windows 10

---

# Project Structure

```
computervision
│
├── 1week
│   ├── ex1_grayscale.py
│   ├── ex2_paintingdraw.py
│   ├── ex3_roiget.py
│   ├── soccer.jpg
│   ├── soccer_Gray.jpg
│   ├── colorbw.jpg
│   ├── paintingdrawsoccer.jpg
│   └── roi.jpg
│
└── README.md
```

---

# Assignment 1 : Grayscale Conversion

## Description

컬러 이미지를 **Grayscale 이미지로 변환**하는 프로그램을 구현한다.  

Grayscale은 RGB 3채널 대신 **밝기 정보만 가진 1채널 이미지**로 변환하는 과정이다.

---

## Key Function

```
cv.imread()
cv.cvtColor()
cv.imshow()
cv.imwrite()
```

---

## Code Explanation

1. `cv.imread()` 로 이미지 불러오기  
2. `cv.cvtColor()` 로 BGR → Grayscale 변환  
3. `cv.imshow()` 로 이미지 출력  
4. `cv.imwrite()` 로 결과 이미지 저장  

---

## Result

### Original Image

![original](soccer.jpg)

### Grayscale Image

![gray](1week/soccer_Gray.jpg)

### Mixed Image

![Final](1week/colorbw.jpg)

---

# Assignment 2 : Painting Draw

## Description

마우스를 이용하여 **이미지 위에 직접 그림을 그릴 수 있는 프로그램**을 구현한다.

---

## Function

- 좌클릭 드래그 → 파란색 브러시  
- 우클릭 드래그 → 빨간색 브러시  
- 브러시 크기 조절 가능  

---

## Key Function

```
cv.setMouseCallback()
cv.circle()
cv.imshow()
```

---

## Code Explanation

1. `cv.setMouseCallback()` 을 이용해 마우스 이벤트 감지  
2. 마우스 클릭 위치 `(x, y)` 좌표를 가져옴  
3. `cv.circle()` 을 이용하여 해당 위치에 원을 그림  
4. `thickness = -1` 옵션으로 **채워진 원 생성**

---

## Result

이미지 위에 자유롭게 그림을 그릴 수 있다.

![Final](1week/paintingdrawsoccer.jpg)

---

# Assignment 3 : ROI (Region of Interest)

## Description

마우스를 이용해 **이미지에서 특정 영역을 선택하고 추출하는 프로그램**을 구현한다.

---

## Function

- 드래그로 ROI 선택  
- 선택한 영역 별도 창 출력  
- `r` 키 → ROI 초기화  
- `s` 키 → ROI 저장  
- `q` 키 → 프로그램 종료  

---

## Key Function

```
cv.setMouseCallback()
cv.rectangle()
numpy slicing
cv.imwrite()
```

---

## Code Explanation

1. 마우스 클릭 시 시작 좌표 저장  
2. 드래그 중 `cv.rectangle()` 로 영역 표시  
3. 마우스를 놓으면 ROI 좌표 확정  
4. NumPy 슬라이싱으로 ROI 추출

```python
roi = img[start_y:y, start_x:x]
```

5. `cv.imwrite()` 로 ROI 이미지 저장

---

## Result

선택한 영역만 추출된 이미지가 새로운 창에 표시된다.

![ROI](1week/roi.jpg)

---

# How to Run

```
python ex1_grayscale.py
python ex2_paintingdraw.py
python ex3_roiget.py
```

---

# Learning Outcome

이번 과제를 통해 다음 내용을 학습하였다.

- OpenCV 기본 이미지 처리
- 마우스 이벤트 처리
- 이미지 ROI 추출
- Python 기반 컴퓨터 비전 기초

---

# Author

Computer Vision Assignment  
Dong-A University
