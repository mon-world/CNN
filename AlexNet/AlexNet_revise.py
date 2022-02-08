import tensorflow.keras as keras
from tensorflow.keras import layers

'''
https://github.com/tensorflow/models/tree/master/research/slim
위 깃허브는 CNN에 모델 코드가 있는 깃허브이다.
이 중 AlexNet 부분을 가져와, 실제로 동작하게 코드를 고치는 작업을 하였다.
(https://github.com/tensorflow/models/blob/master/research/slim/nets/alexnet.py)

1. 주석과 설명들을 지우는 작업
2. Structure(구조)를 보며 코드 수정
3. classes나 classifier_activation의 경우, default 값으로 설정한다.
4. training 함수 등을 keras 라이브러리로 바꾼다.
5. 모델이 시작하는 input과 끝인 model = keras.Model를 찾는다.
6. 코드들의 줄 정리

7. is_training : 케라스에선 자동으로 호출. 학습할떄와 테스트 시의 드랍아웃을 결정한다. 이제는 잘 사용하지 않는다.
8. keras에는 squeeze가 없으므로, 같은 역할을 하는 Flatten을 사용한다.
9. slim의 클래스와 라이브러리를 keras로 바꾼다.
'''

model = keras.Sequential()
model.add(keras.layers.InputLayer([224, 224, 3]))

# 원본 복구하기 위해 5,5 유지를 위해 valid를 사용함.
model.add(keras.layers.Conv2D(64, [11, 11], 4, 'valid', activation='relu'))
model.add(keras.layers.MaxPool2D([3, 3], 2, 'valid'))
model.add(keras.layers.Conv2D(192, [5, 5], 1, 'same', activation='relu'))
model.add(keras.layers.MaxPool2D([3, 3], 2, 'valid'))
model.add(keras.layers.Conv2D(384, [3, 3], 1, 'same', activation='relu'))
model.add(keras.layers.Conv2D(384, [3, 3], 1, 'same', activation='relu'))
model.add(keras.layers.Conv2D(256, [3, 3], 1, 'same', activation='relu'))
model.add(keras.layers.MaxPool2D([3, 3], 2, 'valid'))

model.add(keras.layers.Conv2D(4096, [5, 5], 1, 'valid', activation='relu'))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Conv2D(4096, [1, 1], 1, activation='relu'))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Conv2D(1000, [1, 1], 1))
model.add(layers.Flatten())
model.add(keras.layers.Activation('softmax'))

model.summary()

# Total params: 50,303,912
# Trainable params: 50,303,912
# Non-trainable params: 0
