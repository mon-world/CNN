
import tensorflow.keras as keras
import pandas as pd
import numpy as np

def linear_regression():
    x = [1,2,3]
    y = [1,2,3]

    model = keras.Sequential()
    model.add(keras.layers.Dense(1))                # wx + b

    model.compile(optimizer=keras.optimizers.SGD(0.1),
                  loss = keras.losses.mse,          # mean square error
                  metrics= ['accuracy'])

    model.fit(x,y, epochs=100, verbose= 2)      # 0(없음) 1(전체) 2(약식)
    model.summary()

    print(model.predict(x))

    x = [5,7]
    print(model.predict(x))

def multiple_regression():  # 피쳐가 여러개 있는 경우.
    # 공부시간 출석일수 라고 가정
    x = [[1,2],
         [2,1],
         [4,5],
         [5,4],
         [8,9],
         [9,8]]
    y = [[3],
         [3],
         [9],
         [9],
         [17],
         [17]]

    model = keras.Sequential()
    model.add(keras.layers.Dense(1))                # wx + b

    model.compile(optimizer=keras.optimizers.SGD(0.01),
                  loss = keras.losses.mse,          # mean square error
                  metrics= ['accuracy'])

    model.fit(x,y, epochs=100, verbose= 2)      # 0(없음) 1(전체) 2(약식)

    # 3시간 공부하고 8번 출석한 학생과
    # 6시간 공부하고 2번 출석한 학생의 성적을 구하기
    print(model.predict([[3,8],[6,2]]))


def multiple_regression_trees():
    trees = pd.read_csv("../data/trees.csv")
    print(trees)
    print(trees.values)
    x = trees.values[:, :-1]
    y = trees.values[:, -1:]
    print(x.shape, y.shape)

    model = keras.Sequential()
    model.add(keras.layers.Dense(1))                # wx + b

    model.compile(optimizer=keras.optimizers.SGD(0.0001),
                  loss = keras.losses.mse,          # mean square error
                  metrics= ['accuracy'])

    model.fit(x, y, epochs=100, verbose= 2)      # 0(없음) 1(전체) 2(약식)





linear_regression()
# multiple_regression()
# multiple_regression_trees()