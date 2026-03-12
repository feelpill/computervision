import cv2                     # OpenCV 라이브러리 (이미지 처리)
import numpy as np             # 수치 계산용 NumPy
from pathlib import Path       # 파일 경로 관리용 모듈

# 출력 폴더 생성
output_dir = Path("./outputs")                     # 결과 이미지들을 저장할 폴더 경로 설정
output_dir.mkdir(parents=True, exist_ok=True)      # 폴더가 없으면 생성, 이미 있으면 무시

# 좌/우 이미지 불러오기
left_color = cv2.imread("left.png")                # 왼쪽 카메라 이미지 읽기
right_color = cv2.imread("right.png")              # 오른쪽 카메라 이미지 읽기

# 이미지 로드 실패 검사
if left_color is None or right_color is None:      # 둘 중 하나라도 읽기 실패하면
    raise FileNotFoundError("좌/우 이미지를 찾지 못했습니다.")  # 오류 발생

# -----------------------------
# 카메라 파라미터
# -----------------------------
f = 700.0      # 카메라 초점거리 (pixel 단위)
B = 0.12       # 두 카메라 사이 거리 (baseline, meter)

# -----------------------------
# 관심 영역(ROI) 설정
# x, y = 시작 좌표
# w, h = 너비와 높이
# -----------------------------
rois = {
    "Painting": (55, 50, 130, 110),
    "Frog": (90, 265, 230, 95),
    "Teddy": (310, 35, 115, 90)
}

# -----------------------------
# 그레이스케일 변환
# stereo matching은 grayscale에서 수행
# -----------------------------
left_gray = cv2.cvtColor(left_color, cv2.COLOR_BGR2GRAY)    # 왼쪽 이미지 grayscale 변환
right_gray = cv2.cvtColor(right_color, cv2.COLOR_BGR2GRAY)  # 오른쪽 이미지 grayscale 변환

# -----------------------------
# 1. Disparity 계산
# -----------------------------
stereo = cv2.StereoBM_create(numDisparities=64, blockSize=15)
# StereoBM 알고리즘 생성
# numDisparities → disparity 탐색 범위
# blockSize → 블록 매칭에 사용할 패치 크기

disparity = stereo.compute(left_gray, right_gray).astype(np.float32) / 16.0
# 좌우 이미지의 disparity 계산
# OpenCV StereoBM은 disparity 값을 16배로 저장하므로 16으로 나눔

# -----------------------------
# 2. Depth 계산
# Z = fB / d
# -----------------------------
depth_map = np.zeros_like(disparity, dtype=np.float32)   # depth 결과 저장 배열 생성
valid_mask = disparity > 0                               # disparity가 유효한 위치

depth_map[valid_mask] = (f * B) / disparity[valid_mask]
# 깊이 계산 공식
# Z = fB / d
# f → 초점거리
# B → 카메라 baseline
# d → disparity

# -----------------------------
# 3. ROI별 평균 disparity / depth 계산
# -----------------------------
results = {}     # 결과 저장 딕셔너리

for name, (x, y, w, h) in rois.items():   # ROI 하나씩 처리

    roi_disp = disparity[y:y+h, x:x+w]    # ROI 영역 disparity 추출
    roi_depth = depth_map[y:y+h, x:x+w]   # ROI 영역 depth 추출

    valid_roi = roi_disp > 0              # 유효 disparity만 선택

    if np.any(valid_roi):                 # 유효 값이 존재하면
        mean_disp = np.mean(roi_disp[valid_roi])    # 평균 disparity
        mean_depth = np.mean(roi_depth[valid_roi])  # 평균 depth
    else:
        mean_disp = np.nan
        mean_depth = np.nan

    results[name] = (mean_disp, mean_depth)  # 결과 저장

# -----------------------------
# 4. 결과 출력
# -----------------------------
print("\nROI 평균 Disparity / Depth")

for name, (d, z) in results.items():      # ROI별 출력
    print(f"{name}: disparity={d:.2f}, depth={z:.3f} m")

# disparity가 클수록 가까움
closest = max(results.items(), key=lambda x: x[1][0])[0]   # 가장 가까운 객체
farthest = min(results.items(), key=lambda x: x[1][0])[0]  # 가장 먼 객체

print("\n가장 가까운 ROI:", closest)
print("가장 먼 ROI:", farthest)

# -----------------------------
# 5. disparity 시각화
# 가까울수록 빨강 / 멀수록 파랑
# -----------------------------
disp_tmp = disparity.copy()     # disparity 복사
disp_tmp[disp_tmp <= 0] = np.nan   # invalid disparity 제거

if np.all(np.isnan(disp_tmp)):     # 모든 값이 invalid면
    raise ValueError("유효한 disparity 값이 없습니다.")

# outlier 제거용 percentile
d_min = np.nanpercentile(disp_tmp, 5)
d_max = np.nanpercentile(disp_tmp, 95)

if d_max <= d_min:                 # 값이 같으면 오류 방지
    d_max = d_min + 1e-6

disp_scaled = (disp_tmp - d_min) / (d_max - d_min)   # 0~1 정규화
disp_scaled = np.clip(disp_scaled, 0, 1)

disp_vis = np.zeros_like(disparity, dtype=np.uint8)  # 시각화용 이미지
valid_disp = ~np.isnan(disp_tmp)

disp_vis[valid_disp] = (disp_scaled[valid_disp] * 255).astype(np.uint8)
# 0~255 범위로 변환

disparity_color = cv2.applyColorMap(disp_vis, cv2.COLORMAP_JET)
# 컬러맵 적용
# JET: 빨강(가까움) → 파랑(멀음)

# -----------------------------
# 6. depth 시각화
# 가까울수록 빨강 / 멀수록 파랑
# -----------------------------
depth_vis = np.zeros_like(depth_map, dtype=np.uint8)

if np.any(valid_mask):   # 유효 depth 존재 시

    depth_valid = depth_map[valid_mask]

    z_min = np.percentile(depth_valid, 5)
    z_max = np.percentile(depth_valid, 95)

    if z_max <= z_min:
        z_max = z_min + 1e-6

    depth_scaled = (depth_map - z_min) / (z_max - z_min)
    depth_scaled = np.clip(depth_scaled, 0, 1)

    depth_scaled = 1.0 - depth_scaled
    # depth는 가까울수록 값이 작기 때문에
    # 색상 반전을 위해 뒤집음

    depth_vis[valid_mask] = (depth_scaled[valid_mask] * 255).astype(np.uint8)

depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
# depth 컬러맵 적용

# -----------------------------
# 7. Left / Right 이미지에 ROI 표시
# -----------------------------
left_vis = left_color.copy()    # 원본 복사
right_vis = right_color.copy()

for name, (x, y, w, h) in rois.items():

    cv2.rectangle(left_vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
    # 왼쪽 이미지 ROI 박스 그리기

    cv2.putText(left_vis, name, (x, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    # ROI 이름 표시

    cv2.rectangle(right_vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
    # 오른쪽 이미지 ROI 박스

    cv2.putText(right_vis, name, (x, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

# -----------------------------
# 8. 저장
# -----------------------------
cv2.imwrite(str(output_dir / "disparity_color.png"), disparity_color)
# disparity 컬러맵 저장

cv2.imwrite(str(output_dir / "depth_color.png"), depth_color)
# depth 컬러맵 저장

cv2.imwrite(str(output_dir / "left_roi.png"), left_vis)
# ROI 표시된 left 이미지 저장

cv2.imwrite(str(output_dir / "right_roi.png"), right_vis)
# ROI 표시된 right 이미지 저장

# -----------------------------
# 9. 출력
# -----------------------------
cv2.imshow("Left ROI", left_vis)         # left 이미지 출력
cv2.imshow("Right ROI", right_vis)       # right 이미지 출력
cv2.imshow("Disparity", disparity_color) # disparity 시각화
cv2.imshow("Depth", depth_color)         # depth 시각화

cv2.waitKey(0)            # 키 입력 대기
cv2.destroyAllWindows()   # 모든 창 닫기