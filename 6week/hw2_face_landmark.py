"""
L06 과제2: Mediapipe를 활용한 얼굴 랜드마크 추출 및 시각화
- Mediapipe FaceLandmarker(Tasks API)로 468개 얼굴 랜드마크 검출
- OpenCV 웹캠 실시간 영상에 랜드마크 점 표시
- ESC 키로 종료
"""

import os
import urllib.request
import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    FaceLandmarker,
    FaceLandmarkerOptions,
    RunningMode,
)


MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "face_landmarker.task")


def download_model():
    """모델 파일이 없으면 다운로드"""
    if not os.path.exists(MODEL_PATH):
        print("모델 다운로드 중...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("모델 다운로드 완료.")


def main():
    download_model()

    # 결과 저장용 변수 (콜백에서 갱신)
    latest_result = [None]

    def on_result(result, output_image, timestamp_ms):
        latest_result[0] = result

    # FaceLandmarker 초기화 (LIVE_STREAM 모드)
    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=RunningMode.LIVE_STREAM,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        result_callback=on_result,
    )
    landmarker = FaceLandmarker.create_from_options(options)

    # 웹캠 캡처
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: 웹캠을 열 수 없습니다.")
        return

    print("얼굴 랜드마크 검출 실행 중... (ESC: 종료)")
    frame_timestamp = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 좌우 반전 (거울 효과)
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        # Mediapipe Image 변환 후 비동기 검출
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        landmarker.detect_async(mp_image, frame_timestamp)
        frame_timestamp += 33  # ~30fps 간격

        # 랜드마크 시각화
        result = latest_result[0]
        if result and result.face_landmarks:
            for face_landmarks in result.face_landmarks:
                for landmark in face_landmarks:
                    # 정규화 좌표 -> 픽셀 좌표 변환
                    x = int(landmark.x * w)
                    y = int(landmark.y * h)
                    cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

        cv2.imshow("Face Landmark Detection", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()
    print("프로그램 종료.")


if __name__ == "__main__":
    main()
