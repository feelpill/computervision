import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False

# 1. MNIST 데이터셋 로드
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# 2. 데이터 전처리 (0~1 정규화)
x_train = x_train / 255.0
x_test = x_test / 255.0

print(f"훈련 데이터: {x_train.shape}, 테스트 데이터: {x_test.shape}")

# 3. 간단한 신경망 모델 구축
model = Sequential([
    Flatten(input_shape=(28, 28)),       # 28x28 이미지를 784 벡터로 변환
    Dense(128, activation='relu'),        # 은닉층 128개 뉴런
    Dense(64, activation='relu'),         # 은닉층 64개 뉴런
    Dense(10, activation='softmax')       # 출력층 10개 클래스 (0~9)
])

model.summary()

# 4. 모델 컴파일
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 5. 모델 훈련
history = model.fit(x_train, y_train, epochs=5, batch_size=32,
                    validation_split=0.1)

# 6. 모델 평가
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"\n테스트 정확도: {test_acc:.4f}")

# 7. 예측 결과 시각화
predictions = model.predict(x_test[:5])

plt.figure(figsize=(12, 3))
for i in range(5):
    plt.subplot(1, 5, i + 1)
    plt.imshow(x_test[i], cmap='gray')
    plt.title(f"예측: {predictions[i].argmax()}\n정답: {y_test[i]}")
    plt.axis('off')
plt.suptitle(f"MNIST 분류 결과 (테스트 정확도: {test_acc:.4f})")
plt.tight_layout()
plt.savefig('01_mnist_result.png', dpi=150)
plt.show()
