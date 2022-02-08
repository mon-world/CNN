# softmax 사용해보기
import tensorflow.keras as keras
import pandas as pd
import numpy as np
from sklearn import preprocessing, model_selection

# activation 함수를 softmax로 적용
def softmax_regression():
    # 공부시간 출석일수
    x = [[1, 2],        # C
         [2, 1],
         [4, 5],        # B
         [5, 4],
         [8, 9],        # A
         [9, 8]]
    y = [[0, 0, 1],     # one-hot vector
         [0, 0, 1],
         [0, 1, 0],
         [0, 1, 0],
         [1, 0, 0],
         [1, 0, 0]]

    # 0~2로 채운다면, 숫자로 인식하기 때문에, 결과가 정확하게 나오기 힘들다.
    # 원 핫 벡터를 통해 구분지어줄 수 있다.

    model = keras.Sequential()
    model.add(keras.layers.Dense(3, activation=keras.activations.softmax))

    model.compile(optimizer=keras.optimizers.SGD(0.1),
                  loss=keras.losses.categorical_crossentropy,
                  metrics='acc')

    model.fit(x, y, epochs=20, verbose=2)
    print(model.evaluate(x, y, verbose=0))

    p = model.predict(x)
    print(p)

    p_arg = np.argmax(p, axis=1)
    y_arg = np.argmax(y, axis=1)
    print(p_arg)
    print(y_arg)

    print('acc :', np.mean(p_arg == y_arg))


# x, y 데이터 셋에 대해 7대 3으로 나누어서
# 학습 데이터로 학습하고 검사 데이터에 대해 정확도 구하기
def softmax_regression_iris():
    iris = pd.read_csv('../data/iris_onehot.csv')
    # print(iris)

    x = iris.values[:, :-3]
    y = iris.values[:, -3:]
    # print(x.shape, y.shape)               # (150, 4) (150, 3)

    # 틀린 코트(정확도 0%, 검사 데이터에 Virginica만 포함)
    # train_size = int(len(x) * 0.7)
    # x_train, x_test = x[:train_size], x[train_size:]
    # y_train, y_test = y[:train_size], y[train_size:]
    # print(x_train.shape, x_test.shape)    # (105, 4) (45, 4)
    # print(y_train.shape, y_test.shape)    # (105, 3) (45, 3)

    # 75대 25로 분할
    # x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y)

    # 셔플이 기본
    x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, train_size=0.7)

    model = keras.Sequential()
    model.add(keras.layers.Dense(3, activation=keras.activations.softmax))

    model.compile(optimizer=keras.optimizers.SGD(0.01),
                  loss=keras.losses.categorical_crossentropy,
                  metrics='acc')

    model.fit(x_train, y_train, epochs=100, verbose=2)
    print(model.evaluate(x_test, y_test, verbose=0))

    p = model.predict(x_test)

    p_arg = np.argmax(p, axis=1)
    y_arg = np.argmax(y_test, axis=1)
    print(y_arg)

    print('acc :', np.mean(p_arg == y_arg))

# 데이터를 onehot 인코딩을 통해 데이터 전처리 하기
def softmax_regression_iris_onehot():
    iris = pd.read_csv('../data/iris.csv')
    # print(iris)

    x = iris.values[:, :-1]
    y = iris.values[:, -1:]
    # print(x.shape, y.shape)               # (150, 4) (150, 1)

    # 원 핫 인코딩
    lb = preprocessing.LabelBinarizer()
    y = lb.fit_transform(y)
    # print(y.shape)                        # (150, 3)
    # print(y[:3])                          # [[1 0 0] [1 0 0] [1 0 0]]

    x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, train_size=0.7)

    model = keras.Sequential()
    model.add(keras.layers.Dense(3, activation=keras.activations.softmax))

    model.compile(optimizer=keras.optimizers.SGD(0.01),
                  loss=keras.losses.categorical_crossentropy,
                  metrics='acc')

    model.fit(x_train, y_train, epochs=100, verbose=2)
    print(model.evaluate(x_test, y_test, verbose=0))

    # p = model.predict(x_test)
    #
    # p_arg = np.argmax(p, axis=1)
    # y_arg = np.argmax(y_test, axis=1)
    #
    # print('acc :', np.mean(p_arg == y_arg))


# softmax_regression()
# softmax_regression_iris()
softmax_regression_iris_onehot()







