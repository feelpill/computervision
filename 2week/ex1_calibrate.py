import cv2                    # OpenCV 라이브러리 (영상 처리)
import numpy as np           # 수치 계산용 numpy
import glob                  # 파일 경로 검색용 (이번 코드에서는 사실상 사용 안됨)

CHECKERBOARD = (9, 6)        # 체크보드 내부 코너 개수 (가로 9개, 세로 6개)

square_size = 25.0           # 체크보드 한 칸의 실제 크기 (mm 단위)

criteria = (                 # 코너 정밀화(cornerSubPix)의 종료 조건 설정
    cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,  # 반복 횟수 + 정확도 기준
    30,                      # 최대 반복 횟수
    0.001                    # 코너 위치 변화가 0.001 이하이면 종료
)

objp = np.zeros((CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)   # 3D 실제 좌표 배열 생성 (Z=0 평면)

objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)  
# 체크보드 코너의 x,y 좌표 생성 (격자 형태)

objp *= square_size          # 실제 체크보드 칸 크기를 반영하여 mm 단위 좌표 생성

objpoints = []               # 여러 이미지에서 얻은 실제 세계 좌표 저장 리스트
imgpoints = []               # 여러 이미지에서 검출된 이미지 좌표 저장 리스트

images = []                  # 사용할 이미지 파일 목록 저장
for i in range(1, 14):
    images.append(f"calibration_images/left{i:02d}.jpg")  
    # left01.jpg ~ left13.jpg 파일 경로 생성

img_size = None              # 이미지 크기 저장 변수

# -----------------------------
# 1. 체크보드 코너 검출
# -----------------------------

for fname in images:         # 이미지 목록을 하나씩 처리

    img = cv2.imread(fname)  # 이미지 읽기

    if img is None:          # 이미지가 없으면
        print("이미지 못 읽음:", fname)  # 오류 메시지 출력
        continue             # 다음 이미지로 넘어감

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  
    # 코너 검출은 grayscale 이미지에서 수행

    if img_size is None:
        img_size = gray.shape[::-1]  # 이미지 크기 저장 (width, height)

    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)  
    # 체크보드 코너 찾기 (성공 여부 ret, 코너 좌표 corners)

    if ret:                  # 코너 검출 성공 시

        objpoints.append(objp)   # 실제 세계 좌표 저장

        corners2 = cv2.cornerSubPix(   # 코너 위치를 서브픽셀 수준으로 정밀화
            gray,
            corners,
            (11,11),                  # 탐색 윈도우 크기
            (-1,-1),                  # 중앙 기준 자동 설정
            criteria                  # 종료 조건
        )

        imgpoints.append(corners2)   # 정밀화된 코너 좌표 저장

        cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)  
        # 이미지 위에 검출된 코너 시각화

        cv2.imshow("Corners", img)   # 코너 검출 결과 화면 표시
        cv2.waitKey(200)             # 0.2초 동안 표시

    else:
        print("코너 검출 실패:", fname)   # 코너 검출 실패 메시지

cv2.destroyAllWindows()      # 모든 OpenCV 창 닫기

# -----------------------------
# 2. 카메라 캘리브레이션
# -----------------------------

ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(  # 카메라 파라미터 계산
    objpoints,       # 실제 세계 좌표
    imgpoints,       # 이미지 좌표
    img_size,        # 이미지 크기
    None,            # 초기 camera matrix (없으면 자동 계산)
    None             # 초기 distortion (없으면 자동 계산)
)

print("Camera Matrix K:")   # 카메라 내부 파라미터 출력
print(K)

print("\nDistortion Coefficients:")  # 렌즈 왜곡 계수 출력
print(dist)

# -----------------------------
# 3. Reprojection Error 계산
# -----------------------------

total_error = 0   # 전체 reprojection error 누적 변수

for i in range(len(objpoints)):   # 각 이미지에 대해 반복

    imgpoints2, _ = cv2.projectPoints(   # 3D 점을 다시 이미지로 투영
        objpoints[i],
        rvecs[i],
        tvecs[i],
        K,
        dist
    )

    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)  
    # 실제 코너와 재투영된 코너 사이의 거리 계산

    total_error += error   # 오차 누적

print("\nMean Reprojection Error:", total_error/len(objpoints))  
# 평균 reprojection error 출력 (캘리브레이션 정확도)

# -----------------------------
# 4. 왜곡 보정 시각화
# -----------------------------

img = cv2.imread(images[0])   # 첫 번째 이미지 사용

h, w = img.shape[:2]          # 이미지 높이와 너비

newcameramtx, roi = cv2.getOptimalNewCameraMatrix(  # 왜곡 보정을 위한 최적 카메라 행렬 계산
    K,
    dist,
    (w,h),
    1,
    (w,h)
)

undistorted = cv2.undistort(  # 이미지 왜곡 보정
    img,
    K,
    dist,
    None,
    newcameramtx
)

cv2.imshow("Original", img)        # 원본 이미지 표시
cv2.imshow("Undistorted", undistorted)  # 왜곡 보정된 이미지 표시

cv2.imwrite("undistorted_result.jpg", undistorted)  
# 왜곡 보정 결과 이미지 저장

cv2.waitKey(0)   # 키 입력 대기
cv2.destroyAllWindows()   # 창 닫기