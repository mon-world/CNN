# linnerud 데이터셋에 대해 동작하는 모델을 구축하기.
from sklearn import datasets, preprocessing
import tensorflow.keras as keras
import numpy as np

# 인풋 3개, 아웃풋 3개인 모델 구축
def innerud():

    x, y = datasets.load_linnerud(return_X_y=True)

    # minmax는 0~1 사이로 바꾸기 때문에, 전체적 loss가 줄어든다.
    # x = preprocessing.scale(x)
    x = preprocessing.minmax_scale(x)
    x1 = x[:, 0:1]
    x2 = x[:, 1:2]
    x3 = x[:, 2:3]
    y1 = y[:, 0:1].reshape(-1)
    y2 = y[:, 1:2].reshape(-1)
    y3 = y[:, 2:3].reshape(-1)

    inputs1 = keras.layers.Input([])
    output1 = keras.layers.Dense(4, activation='relu')(inputs1)
    output1 = keras.layers.Dense(2, activation='relu')(output1)

    inputs2 = keras.layers.Input([1])
    output2 = keras.layers.Dense(4, activation='relu')(inputs2)
    output2 = keras.layers.Dense(2, activation='relu')(output2)

    inputs3 = keras.layers.Input([1])
    output3 = keras.layers.Dense(4, activation='relu')(inputs3)
    output3 = keras.layers.Dense(2, activation='relu')(output3)

    concat = keras.layers.concatenate([output1, output2, output3])

    output4 = keras.layers.Dense(2, activation='relu')(concat)
    output4 = keras.layers.Dense(1, activation='sigmoid')(output4)

    output5 = keras.layers.Dense(2, activation='relu')(concat)
    output5 = keras.layers.Dense(1, activation='sigmoid')(output5)

    output6 = keras.layers.Dense(2, activation='relu')(concat)
    output6 = keras.layers.Dense(1, activation='sigmoid')(output6)

    model = keras.Model([inputs1, inputs2, inputs3], [output4, output5, output6])
    model.summary()

    model.compile(optimizer=keras.optimizers.Adam(0.001),
                  loss=keras.losses.binary_crossentropy,
                  metrics='acc')

    model.fit([x1, x2, x3], [y1, y2, y3], epochs=10, verbose=2)


def innerud_1inputs() :

    # 구분을 위해, branch 마다 이름을 넣어주엇다.
    x, y = datasets.load_linnerud(return_X_y=True)

    # 2개를 모두 해봐야한다.
    # minmax는 0~1 사이로 바꾸기 때문에, 전체적 loss가 줄어든다.
    # x = preprocessing.scale(x)
    x = preprocessing.minmax_scale(x)
    y1 = y[:, 0:1]
    y2 = y[:, 1:2]
    y3 = y[:, 2:3]

    inputs = keras.Input([3])
    output = keras.layers.Dense(12, activation='relu')(inputs)
    output = keras.layers.Dense(6, activation='relu')(output)

    output1 = keras.layers.Dense(4, activation='relu', name='sit_up')(output)
    output1 = keras.layers.Dense(1)(output1)

    output2 = keras.layers.Dense(4, activation='relu', name='chin_up')(output)
    output2 = keras.layers.Dense(1)(output2)

    output3 = keras.layers.Dense(4, activation='relu', name='jumps')(output)
    output3 = keras.layers.Dense(1)(output3)

    model = keras.Model(inputs, [output1, output2, output3])
    model.summary()

    model.compile(optimizer=keras.optimizers.SGD(0.1),
                  loss=keras.losses.mse,
                  metrics='mse')    # fit 함수가 학습하며 결과를 출력할 때, mse가 같이 호출된다.
                                    # Mean Absolute Error

    model.fit(x, [y1, y2, y3], epochs=100, verbose=2)
    print(model.evaluate(x, [y1, y2, y3], verbose=0))

    # 퀴즈
    # predict 함수의 결과에 대해 mae 결과를 보여주세요
    p = model.predict(x)
    print(type(p), len(p))  # <class 'list'> 3
    print(type(p[0]))  # <class 'numpy.ndarray'>
    print(p[0].shape)  # (20, 1)

    p1, p2, p3 = model.predict(x)
    print('mse 1 :', np.mean(np.absolute(p1 - y1)))
    print('mse 2 :', np.mean(np.absolute(p2 - y1)))
    print('mse 3 :', np.mean(np.absolute(p3 - y1)))

    p = np.hstack(p)        # np.hstack([p1,p2,p3])
    print(p.shape)          # (20, 3)
    errors = p - y
    print(errors.shpae)
    print(errors[:3])

    abs_errors = np.abs(errors)
    print(abs_errors[:3])

    print('mse :', np.mean(abs_errors, axis=0))

# innerud()
innerud_1inputs()

# 인풋의 갯수는 관계가 없으므로, 위와 같이가 아닌, output의 갯수만 달라지면 된다.