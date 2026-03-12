import cv2                           # OpenCV 라이브러리 불러오기 (이미지 처리용)

# 이미지 읽기
img = cv2.imread("rose.png")         # rose.png 이미지를 읽어서 img 변수에 저장

# 이미지 크기
h, w = img.shape[:2]                 # 이미지의 높이(h)와 너비(w)를 가져옴

# 중심 좌표
center = (w//2, h//2)                # 이미지 중심 좌표 계산 (회전 기준점)

# 회전 + 스케일 행렬 생성
M = cv2.getRotationMatrix2D(center, 30, 0.8)
# Affine 변환 행렬 생성
# center : 회전 기준점 (이미지 중심)
# 30     : 회전 각도 (시계 반대 방향으로 30도 회전)
# 0.8    : 스케일 값 (이미지를 80% 크기로 축소)

# 평행이동 추가
M[0, 2] += 80                        # x축 방향으로 +80픽셀 이동 (오른쪽으로 이동)
M[1, 2] += -40                       # y축 방향으로 -40픽셀 이동 (위쪽으로 이동)

# Affine 변환 적용
result = cv2.warpAffine(img, M, (w, h))
# warpAffine 함수로 이미지 변환 수행
# img : 원본 이미지
# M   : 회전 + 스케일 + 평행이동이 포함된 변환 행렬
# (w,h) : 출력 이미지 크기

# 결과 출력
cv2.imshow("Result", result)         # 변환된 이미지를 화면에 표시
cv2.waitKey(0)                       # 키 입력이 있을 때까지 창 유지
cv2.destroyAllWindows()              # 모든 OpenCV 창 닫기