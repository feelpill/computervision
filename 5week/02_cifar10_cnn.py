import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False
import numpy as np
from PIL import Image

# CIFAR-10 클래스 이름
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# 1. CIFAR-10 데이터셋 로드
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# 2. 데이터 전처리 (0~1 정규화)
x_train = x_train / 255.0
x_test = x_test / 255.0

print(f"훈련 데이터: {x_train.shape}, 테스트 데이터: {x_test.shape}")

# 3. CNN 모델 설계
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

model.summary()

# 4. 모델 컴파일
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 5. 모델 훈련
history = model.fit(x_train, y_train, epochs=10, batch_size=64,
                    validation_split=0.1)

# 6. 모델 성능 평가
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"\n테스트 정확도: {test_acc:.4f}")

# 7. 테스트 이미지(dog.jpg) 예측
img = Image.open('dog.jpg')
img_resized = img.resize((32, 32))
img_array = np.array(img_resized) / 255.0
img_input = np.expand_dims(img_array, axis=0)

prediction = model.predict(img_input)
predicted_class = class_names[np.argmax(prediction)]
confidence = np.max(prediction) * 100

print(f"\ndog.jpg 예측 결과: {predicted_class} ({confidence:.2f}%)")

# 8. 결과 시각화
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# 원본 이미지
axes[0].imshow(img)
axes[0].set_title(f"dog.jpg 예측: {predicted_class} ({confidence:.1f}%)")
axes[0].axis('off')

# 클래스별 확률
axes[1].barh(class_names, prediction[0])
axes[1].set_xlabel('확률')
axes[1].set_title('클래스별 예측 확률')

plt.suptitle(f"CIFAR-10 CNN 분류 (테스트 정확도: {test_acc:.4f})")
plt.tight_layout()
plt.savefig('02_cifar10_result.png', dpi=150)
plt.show()
