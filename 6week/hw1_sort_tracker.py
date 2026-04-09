"""
L06 과제1: SORT 알고리즘을 활용한 다중 객체 추적기 구현
- YOLOv3로 객체 검출
- SORT(칼만 필터 + 헝가리안 알고리즘)로 다중 객체 추적
- 고유 ID와 경계 상자를 비디오 프레임에 실시간 표시
"""

import cv2
import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment

# ==============================
# 칼만 필터 기반 개별 트래커
# ==============================
class KalmanBoxTracker:
    """바운딩 박스에 대한 칼만 필터 트래커"""
    count = 0

    def __init__(self, bbox):
        # 상태: [x_center, y_center, area, aspect_ratio, vx, vy, va]
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1],
        ])
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
        ])
        self.kf.R[2:, 2:] *= 10.0
        self.kf.P[4:, 4:] *= 1000.0
        self.kf.P *= 10.0
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01

        self.kf.x[:4] = self._bbox_to_z(bbox)
        self.time_since_update = 0
        self.hits = 0
        self.hit_streak = 0
        KalmanBoxTracker.count += 1
        self.id = KalmanBoxTracker.count

    @staticmethod
    def _bbox_to_z(bbox):
        """[x1, y1, x2, y2] -> [cx, cy, area, ratio]"""
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        cx = bbox[0] + w / 2.0
        cy = bbox[1] + h / 2.0
        area = w * h
        ratio = w / float(h) if h > 0 else 0
        return np.array([[cx], [cy], [area], [ratio]])

    @staticmethod
    def _z_to_bbox(z):
        """[cx, cy, area, ratio] -> [x1, y1, x2, y2]"""
        cx, cy, area, ratio = z[0], z[1], z[2], z[3]
        area = max(area, 1)
        w = np.sqrt(area * ratio)
        h = area / w if w > 0 else 0
        return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]).flatten()

    def predict(self):
        if self.kf.x[6] + self.kf.x[2] <= 0:
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.time_since_update += 1
        self.hit_streak = 0 if self.time_since_update > 0 else self.hit_streak
        return self._z_to_bbox(self.kf.x)

    def update(self, bbox):
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(self._bbox_to_z(bbox))

    def get_state(self):
        return self._z_to_bbox(self.kf.x)


# ==============================
# IoU 계산
# ==============================
def iou(bb1, bb2):
    x1 = max(bb1[0], bb2[0])
    y1 = max(bb1[1], bb2[1])
    x2 = min(bb1[2], bb2[2])
    y2 = min(bb1[3], bb2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (bb1[2] - bb1[0]) * (bb1[3] - bb1[1])
    area2 = (bb2[2] - bb2[0]) * (bb2[3] - bb2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0


def iou_matrix(detections, trackers):
    matrix = np.zeros((len(detections), len(trackers)))
    for d, det in enumerate(detections):
        for t, trk in enumerate(trackers):
            matrix[d, t] = iou(det, trk)
    return matrix


# ==============================
# SORT 트래커 클래스
# ==============================
class Sort:
    def __init__(self, max_age=5, min_hits=3, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []

    def update(self, detections):
        # 1) 기존 트래커 예측
        predicted = []
        to_remove = []
        for i, trk in enumerate(self.trackers):
            pos = trk.predict()
            if np.any(np.isnan(pos)):
                to_remove.append(i)
            else:
                predicted.append(pos)
        for i in reversed(to_remove):
            self.trackers.pop(i)

        # 2) 헝가리안 알고리즘으로 매칭
        if len(detections) == 0:
            matched, unmatched_dets, unmatched_trks = [], [], list(range(len(self.trackers)))
        elif len(predicted) == 0:
            matched, unmatched_dets, unmatched_trks = [], list(range(len(detections))), []
        else:
            cost = iou_matrix(detections, predicted)
            row_idx, col_idx = linear_sum_assignment(-cost)

            matched, unmatched_dets, unmatched_trks = [], [], []
            for d in range(len(detections)):
                if d not in row_idx:
                    unmatched_dets.append(d)
            for t in range(len(predicted)):
                if t not in col_idx:
                    unmatched_trks.append(t)
            for r, c in zip(row_idx, col_idx):
                if cost[r, c] < self.iou_threshold:
                    unmatched_dets.append(r)
                    unmatched_trks.append(c)
                else:
                    matched.append((r, c))

        # 3) 매칭된 트래커 업데이트
        for d, t in matched:
            self.trackers[t].update(detections[d])

        # 4) 새 트래커 생성
        for d in unmatched_dets:
            self.trackers.append(KalmanBoxTracker(detections[d]))

        # 5) 오래된 트래커 제거 & 결과 수집
        results = []
        self.trackers = [t for t in self.trackers if t.time_since_update <= self.max_age]
        for trk in self.trackers:
            if trk.hit_streak >= self.min_hits or trk.time_since_update == 0:
                box = trk.get_state()
                results.append(np.append(box, trk.id))

        return np.array(results) if results else np.empty((0, 5))


# ==============================
# YOLOv3 검출기
# ==============================
def load_yolo(cfg_path, weights_path, names_path):
    net = cv2.dnn.readNetFromDarknet(cfg_path, weights_path)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    with open(names_path, "r") as f:
        classes = [line.strip() for line in f if line.strip()]
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]
    return net, classes, output_layers


def detect_objects(frame, net, output_layers, conf_threshold=0.5, nms_threshold=0.4):
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)

    boxes, confidences, class_ids = [], [], []
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > conf_threshold:
                cx, cy, bw, bh = detection[0:4] * np.array([w, h, w, h])
                x1 = int(cx - bw / 2)
                y1 = int(cy - bh / 2)
                x2 = int(cx + bw / 2)
                y2 = int(cy + bh / 2)
                boxes.append([x1, y1, x2, y2])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # NMS 적용
    if boxes:
        nms_boxes = [[b[0], b[1], b[2] - b[0], b[3] - b[1]] for b in boxes]
        indices = cv2.dnn.NMSBoxes(nms_boxes, confidences, conf_threshold, nms_threshold)
        if len(indices) > 0:
            indices = indices.flatten()
            return np.array([boxes[i] for i in indices]), [class_ids[i] for i in indices]
    return np.empty((0, 4)), []


# ==============================
# 메인 실행
# ==============================
def main():
    import os
    base_dir = os.path.dirname(os.path.abspath(__file__))
    l06_dir = os.path.join(base_dir, "L06")

    cfg_path = os.path.join(l06_dir, "yolov3.cfg")
    weights_path = os.path.join(l06_dir, "yolov3.weights")
    names_path = os.path.join(l06_dir, "coco.names")
    video_path = os.path.join(l06_dir, "slow_traffic_small.mp4")

    net, classes, output_layers = load_yolo(cfg_path, weights_path, names_path)
    tracker = Sort(max_age=5, min_hits=3, iou_threshold=0.3)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: 비디오를 열 수 없습니다 - {video_path}")
        return

    # 색상 팔레트 (ID별 고유 색상)
    np.random.seed(42)
    colors = np.random.randint(0, 255, size=(200, 3), dtype=np.uint8)

    print("SORT 다중 객체 추적기 실행 중... (ESC: 종료)")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 객체 검출
        detections, class_ids = detect_objects(frame, net, output_layers)

        # SORT 추적 업데이트
        tracked = tracker.update(detections)

        # 결과 시각화
        for obj in tracked:
            x1, y1, x2, y2, obj_id = obj.astype(int)
            color = colors[obj_id % len(colors)].tolist()
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"ID: {obj_id}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw, y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow("SORT Multi-Object Tracker", frame)
        if cv2.waitKey(30) & 0xFF == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()
    print("추적 종료.")


if __name__ == "__main__":
    main()
