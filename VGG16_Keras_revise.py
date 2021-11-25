
'''
https://github.com/tensorflow/models/tree/master/research/slim
위 깃허브는 CNN에 모델 코드가 있는 깃허브이다.
이 중 VGG16부분을 가져와, 실제로 동작하게 코드를 고치는 작업을 하였다.
(https://github.com/tensorflow/models/blob/master/research/slim/nets/vgg.py)

1. 주석과 설명들을 지우는 작업
2. Structure(구조)를 보며 코드 수정
3. classes나 classifier_activation의 경우, default 값으로 설정한다.
4. training 함수 등 keras 라이브러리로 바꾼다.
5. 모델이 시작하는 input과 끝인 model = keras.Model를 찾는다.
6. 코드들의 줄 정리
'''

from tensorflow.keras import layers
from tensorflow import keras


# keras.어플로 코드 가져와서, 수정함.
# 수정 규칙 : 과감하게 사용하지 않는 것들 삭제.
def vgg16():

    img_input = layers.Input(shape=[224,224,3])
    # Block 1
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    # Classification block
    x = layers.Flatten(name='flatten')(x)
    x = layers.Dense(4096, activation='relu', name='fc1')(x)
    x = layers.Dense(4096, activation='relu', name='fc2')(x)
    x = layers.Dense(1000, activation='softmax',name='predictions')(x)

    model = keras.Model(img_input, x, name='vgg16')
    model.summary()

    # Total params: 138,357,544
    # Trainable params: 138,357,544
    # Non-trainable params: 0
    # VGG16의 파라미터 수가 동일하게 나온다.
vgg16()