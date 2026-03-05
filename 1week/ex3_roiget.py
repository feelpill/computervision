import cv2 as cv
import sys

# OpenCV의 imread() 함수를 사용하여 이미지 파일을 불러옴
img = cv.imread('soccer.jpg')

# 만약 이미지가 정상적으로 불러와지지 않았다면 프로그램 종료
if img is None:
    sys.exit('파일이 존재하지 않습니다.')

# 원본 이미지를 복사하여 별도의 변수에 저장
# ROI 선택이나 리셋 기능에서 원본 상태로 되돌리기 위해 사용
img_copy = img.copy()

# 현재 마우스를 드래그 중인지 여부를 저장하는 변수
drawing = False

# ROI 선택 시작 좌표를 저장할 변수
# 초기값은 의미 없는 값(-1)으로 설정
start_x, start_y = -1, -1

# 선택된 ROI 이미지를 저장할 변수
roi = None


def draw(event, x, y, flags, param):
    # 함수 내부에서 전역 변수들을 사용하기 위해 global 선언
    global start_x, start_y, drawing, img, roi

    # 마우스 왼쪽 버튼을 누른 순간 발생하는 이벤트
    # ROI 선택의 시작 좌표를 저장
    if event == cv.EVENT_LBUTTONDOWN:
        drawing = True             # 드래그 시작 상태로 변경
        start_x, start_y = x, y    # 시작 좌표 저장

    # 마우스를 이동시키는 동안 발생하는 이벤트
    # 드래그 상태일 때 현재 마우스 위치까지 사각형을 계속 그려서 영역을 시각화
    elif event == cv.EVENT_MOUSEMOVE:
        if drawing:
            # 사각형을 계속 새로 그리기 위해 원본 이미지를 다시 복사
            img = img_copy.copy()

            # 시작 좌표와 현재 마우스 좌표 사이에 사각형을 그림
            # (0,255,0)은 BGR 색상으로 초록색
            # 두께는 2픽셀
            cv.rectangle(img, (start_x, start_y), (x, y), (0, 255, 0), 2)

    # 마우스 왼쪽 버튼을 놓았을 때 발생하는 이벤트
    # ROI 선택이 완료되는 순간
    elif event == cv.EVENT_LBUTTONUP:
        drawing = False

        # 사각형을 최종적으로 화면에 표시
        img = img_copy.copy()
        cv.rectangle(img, (start_x, start_y), (x, y), (0, 255, 0), 2)

        # numpy 슬라이싱을 이용하여 ROI 영역을 잘라냄
        # 이미지 배열은 img[세로(y), 가로(x)] 구조이므로
        # y좌표 범위와 x좌표 범위를 이용하여 영역을 추출
        roi = img_copy[start_y:y, start_x:x]

        # ROI 영역이 비어있지 않은 경우 별도의 창에 출력
        if roi.size != 0:
            cv.imshow("ROI", roi)


# "Image"라는 이름의 창 생성
cv.namedWindow('Image')

# 해당 창에서 발생하는 마우스 이벤트를 draw 함수로 전달
cv.setMouseCallback('Image', draw)


# 프로그램이 종료될 때까지 계속 반복되는 메인 루프
while True:

    # 현재 이미지 상태를 창에 출력
    cv.imshow('Image', img)

    # 키보드 입력을 1ms 동안 대기하고 입력값을 저장
    # & 0xFF는 ASCII 코드의 하위 8비트만 사용하기 위한 연산
    key = cv.waitKey(1) & 0xFF

    # r 키를 누르면 ROI 선택을 초기 상태로 리셋
    if key == ord('r'):
        img = img_copy.copy()

    # s 키를 누르면 선택된 ROI 이미지를 파일로 저장
    elif key == ord('s'):
        if roi is not None:
            cv.imwrite('roi.jpg', roi)
            print("ROI 저장 완료")

    # q 키를 누르면 프로그램 종료
    elif key == ord('q'):
        break


# 모든 OpenCV 창을 닫고 프로그램 종료
cv.destroyAllWindows()