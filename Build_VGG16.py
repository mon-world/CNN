import tensorflow.keras as keras

# https://neurohive.io/en/popular-networks/vgg16/
'''
VGG16 모델은 3X3 필터를 다수 사용하여 만든 모델이다.
224, 224, 3 사이즈의 이미지가 들어가며, 출력은 1000 이다.
이를 구현해 보기 위해 위 사이트에서 Architecture를 보고 VGG16을 구현해보았다.

맥스 풀링을 기준으로, 이미지의 사이즈는 절반으로 줄고,
채널의 수는 각각 64, 128, 256, 512, 512이 된다.
채널의 수를 보면 몇개의 필터가 쓰였는지 알 수 있으므로, 각 칸마다 계산하여 구현하였다.

또한 파라미터의 수를 model.summary()를 통해 도출하였다.

vgg16_dense() 함수는 Architecture를 보고 만들어낸 모델이다.
vgg16_conv1x1d() 함수는 dense 레이어를 conv 레이어로 바꿔본 모델이다.

'''
# VGG16 모델을 구축해서 파라미터 갯수가 1억개가 넘는지 확인하기
def vgg16_dense():

    # Block 1
    model = keras.Sequential()
    model.add(keras.layers.InputLayer([224, 224, 3]))
    model.add(keras.layers.Conv2D(filters=64, kernel_size=[3, 3], strides=1, padding='same', activation='relu'))
    model.add(keras.layers.Conv2D(64, 3, 1, 'same', activation='relu'))
    model.add(keras.layers.MaxPool2D(pool_size=[2, 2], strides=[2, 2], padding='same'))

    # Block 2
    model.add(keras.layers.Conv2D(128, 3, 1, 'same', activation='relu'))
    model.add(keras.layers.Conv2D(128, 3, 1, 'same', activation='relu'))
    model.add(keras.layers.MaxPool2D(2, 2))

    # Block 3
    model.add(keras.layers.Conv2D(256, 3, 1, 'same', activation='relu'))
    model.add(keras.layers.Conv2D(256, 3, 1, 'same', activation='relu'))
    model.add(keras.layers.Conv2D(256, 3, 1, 'same', activation='relu'))
    model.add(keras.layers.MaxPool2D(2, 2))

    # Block 4
    model.add(keras.layers.Conv2D(512, 3, 1, 'same', activation='relu'))
    model.add(keras.layers.Conv2D(512, 3, 1, 'same', activation='relu'))
    model.add(keras.layers.Conv2D(512, 3, 1, 'same', activation='relu'))
    model.add(keras.layers.MaxPool2D(2, 2))

    # Block 5
    model.add(keras.layers.Conv2D(512, 3, 1, 'same', activation='relu'))
    model.add(keras.layers.Conv2D(512, 3, 1, 'same', activation='relu'))
    model.add(keras.layers.Conv2D(512, 3, 1, 'same', activation='relu'))
    model.add(keras.layers.MaxPool2D(2, 2))

    # Flatten 함수로 dense 레이어에 연결하기 위해 1차원으로 만들기.
    model.add(keras.layers.Flatten())

    model.add(keras.layers.Dense(4096, activation='relu'))
    model.add(keras.layers.Dense(4096, activation='relu'))
    model.add(keras.layers.Dense(1000, activation='softmax'))

    model.summary()

    # Total params: 138,357,544
    # Trainable params: 138,357,544
    # Non-trainable params: 0

# 마지막에 오는 덴스 레이어들을 1x1 컨볼루션 레이어로 교체하기.
def vgg16_conv1x1d():
    model = keras.Sequential()
    model.add(keras.layers.InputLayer([224, 224, 3]))
    model.add(keras.layers.Conv2D(filters=64, kernel_size=[3, 3], strides=1, padding='same', activation='relu'))
    model.add(keras.layers.Conv2D(64, 3, 1, 'same', activation='relu'))
    model.add(keras.layers.MaxPool2D(pool_size=[2, 2], strides=[2, 2], padding='same'))

    model.add(keras.layers.Conv2D(128, 3, 1, 'same', activation='relu'))
    model.add(keras.layers.Conv2D(128, 3, 1, 'same', activation='relu'))
    model.add(keras.layers.MaxPool2D(2, 2))

    model.add(keras.layers.Conv2D(256, 3, 1, 'same', activation='relu'))
    model.add(keras.layers.Conv2D(256, 3, 1, 'same', activation='relu'))
    model.add(keras.layers.Conv2D(256, 3, 1, 'same', activation='relu'))
    model.add(keras.layers.MaxPool2D(2, 2))

    model.add(keras.layers.Conv2D(512, 3, 1, 'same', activation='relu'))
    model.add(keras.layers.Conv2D(512, 3, 1, 'same', activation='relu'))
    model.add(keras.layers.Conv2D(512, 3, 1, 'same', activation='relu'))
    model.add(keras.layers.MaxPool2D(2, 2))

    model.add(keras.layers.Conv2D(512, 3, 1, 'same', activation='relu'))
    model.add(keras.layers.Conv2D(512, 3, 1, 'same', activation='relu'))
    model.add(keras.layers.Conv2D(512, 3, 1, 'same', activation='relu'))
    model.add(keras.layers.MaxPool2D(2, 2))

    model.add(keras.layers.Conv2D(4096, 7, 1, 'valid', activation='relu'))
    # dense 레이어는 크기가 고정되어있다(여기서는 4096)
    # 반면, conv의 경우, 안에 들어오는 크기 자체를 다양하게 만들 수 있다.
    # 필터를 가지고 슬라이딩 하면서 출력의 크기를 결정하기 때문이다.
    # 따라서 컨볼루션만으로 구성되어 있다면, 입력의 크기가 달라져도 통과할 수 있다.
    # 입력의 크기와 필터의 크기가 같고, padding = valid를 이용해서 슬라이딩 하지 못하게 막는다면,
    # 이것은 컨볼루션 연산이긴 하지만, 곱셈을 단 한번 수행하므로 만들어지는 크기는 1x1 이다.

    # 주의점
    # 단, 문제는 5번째 컨볼루션 블록을 나온 크기가 7x7 이 아니다.
    # 앞에서 어떤 3차원이 넘어오는지 모르지만, 그 3차원 크기에 맞는 필터의 크기를 사용해야한다.
    # 이럴 경우, model.add(keras.layers.GlobalAvgPoolConv2D())를 사용한다.
    # 앞쪽에서 나온 채널의 갯수를 바꾸지 않는다. 즉, 512가 나온다.

    model.add(keras.layers.Conv2D(4096, 1, 1, 'valid', activation='relu'))
    # 이곳의 1x1 필터는 연산량을 줄이는 것이 아니다. 파라미터 갯수가 여기선 동일함.
    model.add(keras.layers.Conv2D(1000, 1, 1, 'valid'))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Softmax())
    model.summary()

    # Total params: 138,357,544
    # Trainable params: 138,357,544
    # Non-trainable params: 0


# vgg16_dense()
vgg16_conv1x1d()
