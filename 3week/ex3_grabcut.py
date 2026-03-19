import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def run_grabcut():
    # 1. 이미지 불러오기 (경로에 공백이 있으므로 파일명을 정확히 입력합니다)
    image_path = r"C:\computervision\3week\coffee cup.jpg"
    img = cv.imread(image_path)

    if img is None:
        print(f"오류: '{image_path}' 이미지를 찾을 수 없습니다.")
        print("파일명의 띄어쓰기('coffee cup.jpg')가 정확한지 확인해 주세요.")
        return

    # GrabCut은 원본 이미지를 훼손하지 않기 위해 복사본을 사용하는 것이 좋습니다.
    original_img = img.copy()

    # 2. GrabCut에 필요한 마스크와 모델 초기화
    # 마스크: 이미지와 동일한 크기의 1채널(흑백) 빈 도화지
    mask = np.zeros(img.shape[:2], np.uint8)
    
    # 힌트: bgdModel(배경 모델)과 fgdModel(전경 모델)을 1x65 크기의 float64 배열로 초기화
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    # 3. 초기 사각형 영역 설정 (x, y, width, height)
    # 💡 주의: 이 값은 가지고 계신 'coffee cup.jpg' 사진의 해상도에 맞춰 조절해야 합니다!
    # 지금은 대략적으로 이미지 크기의 안쪽 부분(여백 50픽셀)을 사각형으로 잡도록 설정했습니다.
    h, w = img.shape[:2]
    rect = (50, 50, w - 100, h - 100) 

    # 4. cv.grabCut()을 사용하여 대화식 분할 수행
    # 인자: 원본이미지, 마스크, 사각형, 배경모델, 전경모델, 반복횟수(보통 5), 모드(사각형 기준)
    cv.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv.GC_INIT_WITH_RECT)

    # 5. 마스크 값을 사용하여 배경 제거
    # GrabCut 실행 후 mask에는 0~3까지의 값이 들어갑니다.
    # 0: 확실한 배경 (cv.GC_BGD)
    # 1: 확실한 전경 (cv.GC_FGD)
    # 2: 아마도 배경 (cv.GC_PR_BGD)
    # 3: 아마도 전경 (cv.GC_PR_FGD)
    
    # 힌트: np.where()를 사용하여 배경(0, 2)은 0으로, 전경(1, 3)은 1로 변경
    mask2 = np.where((mask == cv.GC_BGD) | (mask == cv.GC_PR_BGD), 0, 1).astype('uint8')

    # 6. 마스크를 원본 이미지에 곱하여 객체 추출
    # mask2는 1채널이므로, 3채널인 컬러 이미지와 곱하기 위해 np.newaxis를 사용해 차원을 맞춰줍니다.
    extracted_img = img * mask2[:, :, np.newaxis]

    # 7. Matplotlib를 사용하여 3개의 결과 나란히 시각화
    # BGR -> RGB 변환 (Matplotlib 출력용)
    original_rgb = cv.cvtColor(original_img, cv.COLOR_BGR2RGB)
    extracted_rgb = cv.cvtColor(extracted_img, cv.COLOR_BGR2RGB)
    
    # mask2는 0과 1로 이루어져 있으므로 시각화를 위해 255를 곱해줍니다 (0=검정, 255=흰색)
    mask_display = mask2 * 255

    plt.figure(figsize=(15, 5))

    # 첫 번째: 원본 이미지
    plt.subplot(1, 3, 1)
    plt.imshow(original_rgb)
    # 사각형 영역을 빨간선으로 그려서 보여주면 이해하기 좋습니다.
    x, y, width, height = rect
    plt.gca().add_patch(plt.Rectangle((x, y), width, height, edgecolor='red', facecolor='none', lw=2))
    plt.title('Original Image (with Rect)')
    plt.axis('off')

    # 두 번째: 마스크 이미지
    plt.subplot(1, 3, 2)
    plt.imshow(mask_display, cmap='gray')
    plt.title('GrabCut Mask')
    plt.axis('off')

    # 세 번째: 배경 제거 (객체 추출) 이미지
    plt.subplot(1, 3, 3)
    plt.imshow(extracted_rgb)
    plt.title('Extracted Object')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_grabcut()