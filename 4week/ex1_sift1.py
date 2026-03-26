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
