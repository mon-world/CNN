# Day_01_03_SoftmaxRegression.py
import tensorflow.keras as keras
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import sklearn.preprocessing as preprocessing



def softmax_regression():
    # 공부시간 출석일수 라고 가정
    x = [[1,2],     # C 학점
         [2,1],
         [4,5],     # B 학점
         [5,4],
         [8,9],     # A 학점
         [9,8]]
    y = [[0,0,1],
         [0,0,1],
         [0,1,0],
         [0,1,0],
         [1,0,0],
         [1,0,0]]

    # 0~2로 채운다면, 숫자로 인식하기 때문에, 결과가 정확하게 나오기 힘들다.
    # 원 핫 벡터를 통해 구분지어줄 수 있다.

    model = keras.Sequential()
    model.add(keras.layers.Dense(3, activation=keras.activations.softmax))

    model.compile(optimizer=keras.optimizers.SGD(0.1),
                  loss = keras.losses.categorical_crossentropy,
                  metrics= 'acc')

    model.fit(x,y, epochs=100, verbose= 2)

    p = model.predict(x)
    print(p)

    p_arg = np.argmax(p, axis=1)
    y_arg = np.argmax(y, axis=1)
    print(p_arg)
    print(y_arg)

    # 최종 정확도 계산
    print('acc :', np.mean(p_arg == y_arg))



softmax_regression()
