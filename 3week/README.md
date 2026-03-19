# 📸 Computer Vision 실습: Edge and Region Detection

본 레포지토리는 컴퓨터 비전(Computer Vision)의 핵심 기초인 **에지 검출(Edge Detection)**과 **영역 분할(Region Segmentation)** 알고리즘을 파이썬(Python)과 OpenCV를 활용하여 직접 구현하고 분석한 실습 프로젝트입니다.

## 🛠 기술 스택 및 환경 (Environment)
* **Language:** Python 3.x
* **Libraries:** * `opencv-python` (cv2): 영상 처리 및 알고리즘 적용
  * `numpy`: 행렬 연산 및 마스크 데이터 처리
  * `matplotlib`: 결과 이미지 비교 및 시각화

## 📂 파일 구조 (Directory Structure)
```text
📦 3week
 ┣ 📂 result                 # 각 알고리즘의 실행 결과 캡처 이미지
 ┃ ┣ 📜 ex1_result.jpg
 ┃ ┣ 📜 ex2_result.jpg
 ┃ ┗ 📜 ex3_result.jpg
 ┣ 📜 ex1_sobeledge.py       # 실습 1: Sobel 필터 에지 검출
 ┣ 📜 ex2_houghlines.py      # 실습 2: 허프 변환 직선 검출
 ┣ 📜 ex3_grabcut.py         # 실습 3: GrabCut 객체 추출
 ┣ 📜 edgeDetectionimage.jpg # (실습용 원본 이미지)
 ┣ 📜 dabo.jpg               # (실습용 원본 이미지)
 ┗ 📜 coffee cup.jpg         # (실습용 원본 이미지)

 