import cv2 as cv
import sys

# 이미지 불러오기
img = cv.imread('soccer.jpg')

# 이미지 없으면 종료
if img is None:
    sys.exit('파일이 존재하지 않습니다.')

brush_size = 5  # 초기 붓 크기

# 마우스 이벤트 처리 함수
def draw(event, x, y, flags, param):
    global brush_size

    # 좌클릭 드래그 → 파란색 (이미지 위에다 마우스 위치에 현재 붓 크기로 채워진 원을 그린다 )
    if flags & cv.EVENT_FLAG_LBUTTON:
        cv.circle(img, (x, y), brush_size, (255, 0, 0), -1)

    # 우클릭 드래그 → 빨간색
    if flags & cv.EVENT_FLAG_RBUTTON:
        cv.circle(img, (x, y), brush_size, (0, 0, 255), -1)


cv.namedWindow('Paint')
cv.setMouseCallback('Paint', draw)

#반복문
while True:
    cv.imshow('Paint', img)

    key = cv.waitKey(1) & 0xFF

    # 붓 크기 증가 15까지 +를 누르면 brush_size가 1씩 증가
    if key == ord('+'):
        brush_size = min(15, brush_size + 1)

    # 붓 크기 감소 1까지 -를 누르면 brush_size가 1씩 감소
    elif key == ord('-'):
        brush_size = max(1, brush_size - 1)

    # q 누르면 종료
    elif key == ord('q'):
        break

cv.destroyAllWindows()