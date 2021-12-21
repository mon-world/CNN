# convolution layer 사용하기
import tensorflow.keras as keras


def lenet_5():
    ((x_train, y_train), (x_test, y_test)) = keras.datasets.mnist.load_data()
    # print(x_train.shape, x_test.shape)        # (60000, 28, 28) (10000, 28, 28)
    # print(y_train.shape, y_test.shape)        # (60000,) (10000,)

    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)
    # print(x_train.shape, x_test.shape)        # (60000, 28, 28, 1) (10000, 28, 28, 1)

    # RGB : 0 ~ 255
    x_train = x_train / 255  # minmax scale
    x_test = x_test / 255

    # 인풋 레이어에는 FLATTEN한 1차원을 사용.

    model = keras.Sequential()
    model.add(keras.layers.InputLayer([28, 28, 1]))
    model.add(keras.layers.Conv2D(filters=6, kernel_size=[5, 5], strides=1, padding='same', activation='relu'))
    model.add(keras.layers.MaxPool2D(pool_size=[2, 2], strides=[2, 2], padding='same'))
    model.add(keras.layers.Conv2D(filters=16, kernel_size=[5, 5], strides=1, padding='valid', activation='relu'))
    model.add(keras.layers.MaxPool2D(pool_size=[2, 2], strides=[2, 2], padding='same'))

    model.add(keras.layers.Flatten())
    # model.add(keras.layers.Reshape([-1]))

    model.add(keras.layers.Dense(120, activation='relu'))
    model.add(keras.layers.Dense(84, activation='relu'))
    model.add(keras.layers.Dense(10, activation='softmax'))
    model.summary()

    model.compile(optimizer=keras.optimizers.Adam(0.01),
                  loss=keras.losses.sparse_categorical_crossentropy,
                  metrics='acc')

    model.fit(x_train, y_train, epochs=10, batch_size=100, verbose=2)
    print(model.evaluate(x_test, y_test, verbose=0))


lenet_5()
