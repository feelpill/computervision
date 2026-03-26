# 4주차 - SIFT 특징점 검출 및 이미지 정합

OpenCV의 SIFT 알고리즘을 활용한 특징점 검출, 매칭, 호모그래피 기반 이미지 정합 실습입니다.

---

## 목차

1. [SIFT를 이용한 특징점 검출 및 시각화](#1-sift를-이용한-특징점-검출-및-시각화)
2. [SIFT를 이용한 두 영상 간 특징점 매칭](#2-sift를-이용한-두-영상-간-특징점-매칭)
3. [호모그래피를 이용한 이미지 정합](#3-호모그래피를-이용한-이미지-정합-image-alignment)

---

## 1. SIFT를 이용한 특징점 검출 및 시각화

**파일:** `ex1_sift1.py`

### 알고리즘 설명

**SIFT (Scale-Invariant Feature Transform)** 는 이미지의 크기, 회전, 조명 변화에 강인한 특징점을 검출하는 알고리즘입니다.

| 단계 | 설명 |
|------|------|
| **Scale-Space 생성** | Gaussian 블러를 다양한 스케일로 적용하여 DoG(Difference of Gaussian) 피라미드 구성 |
| **특징점 후보 검출** | DoG 공간에서 극값(local extrema)을 가진 위치를 특징점 후보로 선택 |
| **특징점 정제** | 낮은 대비(contrast)와 엣지 응답이 강한 후보를 제거하여 안정적인 특징점만 유지 |
| **방향 할당** | 각 특징점 주변의 그래디언트 방향 히스토그램을 계산하여 주 방향 부여 → 회전 불변성 확보 |
| **디스크립터 생성** | 특징점 주변 16×16 영역을 4×4 블록으로 나눠 각 블록의 8방향 그래디언트 히스토그램 → 128차원 벡터 |

### 전체 코드

```python
import cv2 as cv
import matplotlib.pyplot as plt

# 이미지 로드
img = cv.imread('images/mot_color70.jpg')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# SIFT 객체 생성 (nfeatures로 특징점 수 제한)
sift = cv.SIFT_create(nfeatures=500)

# 특징점 검출 및 디스크립터 계산
keypoints, descriptors = sift.detectAndCompute(gray, None)

print(f'검출된 특징점 수: {len(keypoints)}')

# 특징점 시각화 (방향과 크기 포함)
img_keypoints = cv.drawKeypoints(
    img, keypoints, None,
    flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
)

# BGR -> RGB 변환
img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
img_kp_rgb = cv.cvtColor(img_keypoints, cv.COLOR_BGR2RGB)

# 나란히 출력
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
axes[0].imshow(img_rgb)
axes[0].set_title('Original Image', fontsize=14)
axes[0].axis('off')

axes[1].imshow(img_kp_rgb)
axes[1].set_title(f'SIFT Keypoints ({len(keypoints)})', fontsize=14)
axes[1].axis('off')

plt.tight_layout()
plt.show()
```

### 결과

![SIFT 특징점 검출 결과](result/result1.png)

---

## 2. SIFT를 이용한 두 영상 간 특징점 매칭

**파일:** `ex2_sift2.py`

### 알고리즘 설명

두 이미지에서 각각 SIFT 특징점과 디스크립터를 추출한 뒤, **BFMatcher**와 **Lowe's Ratio Test**로 신뢰도 높은 대응 쌍을 선별합니다.

#### BFMatcher (Brute-Force Matcher)

- 한 이미지의 모든 디스크립터를 다른 이미지의 모든 디스크립터와 비교 (전수 탐색)
- 거리 척도: **L2 Norm** (유클리드 거리) — SIFT의 실수형 128차원 벡터에 적합

#### knnMatch (k=2)

각 특징점에 대해 가장 가까운 **2개의 후보 매칭**을 반환합니다.

#### Lowe's Ratio Test

```
최근접 거리(m.distance) < 0.75 × 차근접 거리(n.distance)
```

- 1위 매칭이 2위보다 충분히 가까울 때만 유효한 매칭으로 인정
- 임계값 `0.75` → 낮을수록 엄격한 필터링, 높을수록 더 많은 매칭 허용
- 반복 패턴이나 유사 구조에서 발생하는 **오매칭(false positive) 제거**

### 전체 코드

```python
import cv2 as cv
import matplotlib.pyplot as plt

# 두 이미지 로드
img1 = cv.imread('images/mot_color70.jpg')
img2 = cv.imread('images/mot_color83.jpg')

gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

# SIFT 특징점 추출
sift = cv.SIFT_create()
kp1, des1 = sift.detectAndCompute(gray1, None)
kp2, des2 = sift.detectAndCompute(gray2, None)

print(f'이미지1 특징점 수: {len(kp1)}')
print(f'이미지2 특징점 수: {len(kp2)}')

# BFMatcher로 특징점 매칭 (knnMatch + Lowe's ratio test)
bf = cv.BFMatcher(cv.NORM_L2)
matches = bf.knnMatch(des1, des2, k=2)

# Lowe's ratio test로 좋은 매칭점만 선별
good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)

print(f'전체 매칭 수: {len(matches)}')
print(f'선별된 좋은 매칭 수: {len(good_matches)}')

# 매칭 결과 시각화 (상위 50개)
good_matches_sorted = sorted(good_matches, key=lambda x: x.distance)
img_matches = cv.drawMatches(
    img1, kp1, img2, kp2,
    good_matches_sorted[:50], None,
    flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)

img_matches_rgb = cv.cvtColor(img_matches, cv.COLOR_BGR2RGB)

plt.figure(figsize=(16, 6))
plt.imshow(img_matches_rgb)
plt.title(f'SIFT Feature Matching (Good matches: {len(good_matches)})', fontsize=14)
plt.axis('off')
plt.tight_layout()
plt.show()
```

### 결과

![SIFT 특징점 매칭 결과](result/result2.png)

---

## 3. 호모그래피를 이용한 이미지 정합 (Image Alignment)

**파일:** `ex3_image_alignment.py`

### 알고리즘 설명

SIFT 매칭으로 얻은 대응점 쌍을 이용해 **호모그래피 행렬**을 추정하고, 이를 통해 두 이미지를 하나의 평면으로 정합합니다.

#### 호모그래피 (Homography)

같은 평면을 다른 시점에서 촬영한 두 이미지 간의 **투영 변환(Projective Transformation)** 을 나타내는 3×3 행렬입니다.

```
[x']     [h11 h12 h13]   [x]
[y']  =  [h21 h22 h23] × [y]
[w']     [h31 h32  1 ]   [1]
```

#### RANSAC (Random Sample Consensus)

오매칭(outlier)이 섞인 대응점에서 신뢰할 수 있는 호모그래피를 추정하는 방법입니다.

| 단계 | 설명 |
|------|------|
| **샘플링** | 대응점 중 최소 4쌍을 무작위 선택 |
| **모델 추정** | 선택한 4쌍으로 호모그래피 행렬 계산 |
| **검증** | 모든 대응점에 행렬 적용 → 재투영 오차가 임계값(5px) 이하인 점을 **inlier**로 분류 |
| **반복** | inlier가 가장 많은 모델 선택 |

#### warpPerspective

추정된 호모그래피 행렬로 img1을 변환하여 img2의 시점에 맞게 정렬합니다.

### 전체 코드

```python
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# 두 이미지 로드 (img1.jpg, img2.jpg 선택)
img1 = cv.imread('images/img1.jpg')
img2 = cv.imread('images/img2.jpg')

gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

# SIFT 특징점 검출
sift = cv.SIFT_create()
kp1, des1 = sift.detectAndCompute(gray1, None)
kp2, des2 = sift.detectAndCompute(gray2, None)

print(f'이미지1 특징점 수: {len(kp1)}')
print(f'이미지2 특징점 수: {len(kp2)}')

# BFMatcher + knnMatch로 매칭
bf = cv.BFMatcher(cv.NORM_L2)
matches = bf.knnMatch(des1, des2, k=2)

# Lowe's ratio test (임계값 0.7)
good_matches = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good_matches.append(m)

print(f'선별된 좋은 매칭 수: {len(good_matches)}')

# 호모그래피 계산을 위한 대응점 추출
src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

# RANSAC을 이용한 호모그래피 행렬 계산
H, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
inliers = mask.ravel().sum()
print(f'Inlier 매칭 수: {inliers}')

# 파노라마 크기로 warpPerspective 적용
h1, w1 = img1.shape[:2]
h2, w2 = img2.shape[:2]
output_size = (w1 + w2, max(h1, h2))

warped = cv.warpPerspective(img1, H, output_size)

# img2를 warped 이미지 왼쪽에 합성
result = warped.copy()
result[0:h2, 0:w2] = img2

# 매칭 결과 시각화 (inlier만)
matchesMask = mask.ravel().tolist()
draw_params = dict(
    matchColor=(0, 255, 0),
    singlePointColor=None,
    matchesMask=matchesMask,
    flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)
img_matches = cv.drawMatches(img1, kp1, img2, kp2, good_matches, None, **draw_params)

# BGR -> RGB 변환
result_rgb = cv.cvtColor(result, cv.COLOR_BGR2RGB)
img_matches_rgb = cv.cvtColor(img_matches, cv.COLOR_BGR2RGB)

# 나란히 출력
fig, axes = plt.subplots(1, 2, figsize=(18, 6))
axes[0].imshow(img_matches_rgb)
axes[0].set_title(f'Matching Result (Inliers: {inliers})', fontsize=13)
axes[0].axis('off')

axes[1].imshow(result_rgb)
axes[1].set_title('Warped Image (Image Alignment)', fontsize=13)
axes[1].axis('off')

plt.tight_layout()
plt.show()
```

### 결과

![이미지 정합 결과](result/result3.png)

---

## 실행 환경

```
Python 3.x
opencv-python
numpy
matplotlib
```

## 파일 구조

```
4week/
├── images/
│   ├── mot_color70.jpg   # ex1, ex2 입력 이미지
│   ├── mot_color83.jpg   # ex2 입력 이미지
│   ├── img1.jpg          # ex3 입력 이미지 (선택)
│   ├── img2.jpg          # ex3 입력 이미지 (선택)
│   └── img3.jpg          # ex3 입력 이미지 (선택)
├── result/
│   ├── result1.png       # 특징점 검출 결과
│   ├── result2.png       # 특징점 매칭 결과
│   └── result3.png       # 이미지 정합 결과
├── ex1_sift1.py
├── ex2_sift2.py
├── ex3_image_alignment.py
└── README.md
```
