import cv2 as cv
import sys
import numpy as np

#이미지 파일 읽기
img = cv.imread('soccer.jpg')

#만약 이미지가 없다면 프로그램 종료
if img is None :
    sys.exit('파일이 존재하지 않습니다.')

# 컬러를 그레이(흑백)으로 스케일링해서 변환시킴
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# 그레이로 바꾼 이미지를 soccer_Gray.jpg 파일로 저장
cv.imwrite('soccer_Gray.jpg', gray)

# 화면에 맞게 이미지 크기 줄이기
img_small = cv.resize(img, (640, 480))
gray_small = cv.resize(gray, (640, 480))

# gray는 2차원이라서 읽지를 못해 bgr 3차원으로 다시 변환시킴
gray_bgr = cv.cvtColor(gray_small, cv.COLOR_GRAY2BGR)

# 원본과 흑백을 나란히 연결
mixed = np.hstack((img_small, gray_bgr))

# 결과 보여주기
cv.imshow('colorbw.jpg',mixed)
# 결과 저장
cv.imwrite('colorbw.jpg', mixed)
#아무거나 누르면 종료
cv.waitKey()
cv.destroyAllWindows()
