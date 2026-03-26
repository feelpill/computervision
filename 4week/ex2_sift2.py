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
