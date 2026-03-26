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
