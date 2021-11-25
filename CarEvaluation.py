'''
UCI 머신러닝 데이터넷으로 Car_Evaluation을 예측해보았다.
데이터 셋 주소 : https://archive.ics.uci.edu/ml/datasets/Car+Evaluation

함수의 이름은 car_model_sparse 이라고 하였고, 다음 과정을 통해 설계하였다.
1. 판다스로 파일 읽기                 : header가 없기 때문에, car.name에 항목별로 컬럼 이름을 정해줬다.
2. 문자열을 숫자열로 전처리            : preprocessing.LabelBinarizer를 사용해 원 핫 벡터로 만들어줬다.
3. 전처리 한 입력값들을 합쳐주기        : np.hstack으로 입력 데이터들을 합쳤다.
4. 학습데이터와 시험 데이터로 나누기     : model_selection.train_test_split를 사용하였다.
    주의) car.data의 경우, 무작위 순서가 아닌 일정하게 정렬되어 있으므로, 랜덤하게 섞어줄 필요가 있다.
5. 모델 구축                        : fully connected layer로 구성하였다.
    주의) 이미 원 핫 인코딩이 되어있으므로, categorical_crossentropy를 손실함수로 선택한다.

'''

import tensorflow.keras as keras
import pandas as pd
import numpy as np
from sklearn import preprocessing, model_selection


def car_model_sparse():
    # 1. 판다스로 파일 읽기
    cars = pd.read_csv('data/car.data',
                       header=None,
                       names=['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class'])

    # 2. 문자열을 숫자열로 전처리
    lb = preprocessing.LabelBinarizer()
    buying = lb.fit_transform(cars['buying'])
    maint = lb.fit_transform(cars['maint'])
    doors = lb.fit_transform(cars['doors'])
    persons = lb.fit_transform(cars['persons'])
    lug_boot = lb.fit_transform(cars['lug_boot'])
    safety = lb.fit_transform(cars['safety'])
    y = lb.fit_transform(cars['class'])

    # 3. 전처리 한 입력값들을 합쳐주기
    x = np.hstack([buying, maint, doors, persons, lug_boot, safety])
    print(x.shape)  # (1727, 21)
    print(y.shape)  # (1727, 4)

    # 4. 학습데이터와 시험 데이터로 나누기
    # model_selection을 사용하면 셔플의 개념이 있다.
    x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, train_size=0.7)

    # 5. 모델 구축
    model = keras.Sequential()
    model.add(keras.layers.InputLayer([21]))
    model.add(keras.layers.Dense(256, activation='relu'))
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dense(4, activation='softmax'))

    model.compile(optimizer=keras.optimizers.Adam(0.01),
                  loss=keras.losses.categorical_crossentropy,
                  metrics='acc')

    model.fit(x_train, y_train, epochs=10, batch_size=32, verbose=2)
    print(model.evaluate(x_test, y_test, verbose=0))


car_model_sparse()
# [0.009311912581324577, 0.9961464405059814]
