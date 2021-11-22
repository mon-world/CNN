# Day_01_02_LogisticRegression.py
import tensorflow.keras as keras
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import sklearn.preprocessing as preprocessing

# 딥러닝 : 리그레션(회귀), 클래시피케이션(분류)
# 회귀 : 수치 예측에 사용.
# 분류 : 구분에 사용.
# cnn은 모든게 분류이므로 멀티플 리그레이션 사용 x


def logistic_regression():
    # 공부시간 출석일수 라고 가정
    x = [[1,2],     # 탈락
         [2,1],
         [4,5],     # 통과
         [5,4],
         [8,9],
         [9,8]]
    y = [[0],
         [0],
         [1],
         [1],
         [1],
         [1]]

    model = keras.Sequential()
    model.add(keras.layers.Dense(1, activation=keras.activations.sigmoid))

    model.compile(optimizer=keras.optimizers.SGD(0.1),
                  loss = keras.losses.binary_crossentropy,
                  metrics= 'acc')       # classification은 이걸 사용.

    model.fit(x,y, epochs=100, verbose= 2)

    p = model.predict(x)
    print(p)

    p_bool = (p > 0.5)
    print(p_bool)

    equals = (p_bool == y)
    print(equals)

    # 최종 정확도 계산
    print('acc :', np.mean(equals))

def logistic_regression_pima():
    pima = pd.read_csv('data/pima-indians-diabetes.csv', skiprows=9,
                    header= None) # header가 무조건 있다고 생각하는데,
                                    # 없다고 알려주는 것.
    pima.info()
    print(pima)
    x = pima.values[:, :-1]
    y = pima.values[:, -1:]
    print(x.shape, y.shape)

    x = preprocessing.scale(x)

    # 앞에서 만든 x, y 데이터 셋에 대해 7대 3으로 나누어서
    # 학습 데이터로 학습하고 검사 데이터에 대해 정확도 구하기

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3)
    print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

    # 정확도가 70퍼센트 넘는 모델 만들기

    model = keras.Sequential()
    model.add(keras.layers.Dense(1, activation=keras.activations.sigmoid))
    model.compile(optimizer=keras.optimizers.SGD(0.01),
                  loss = keras.losses.binary_crossentropy,
                  metrics= 'acc')       # classification은 이걸 사용.

    model.fit(x_train, y_train, epochs=200, batch_size= 64, verbose=2)

    p = model.predict(x_test)
    p_bool = (p > 0.5)

    # 최종 정확도 계산
    print('acc :', np.mean(p_bool == y_test))

# logistic_regression
logistic_regression_pima()