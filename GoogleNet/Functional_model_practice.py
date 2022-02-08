import tensorflow.keras as keras
import numpy as np

'''
딥러닝 모델은 sequential형과 functional 형이 있다.
간단한 and 연산을 계산하는 딥러닝 모델을 각각 두 형태로 구현하였다.

또한, 함수형 모델의 특징인 다중 입력, 다중 출력을 구현하였다.

and_sequential()는 Sequential 형태로 AND 연산을 구현하였다.
and_functional()은 Functional 형태로 AND 연산을 구현하였다.
functional_input()은 Functional 형태에서 AND 연산과 다중 입력을 구현하였다.
functional_input_output()은 Functional 형태에서 XOR 연산과 다중입력, 다중 출력을 구현하였다.

'''

# AND 데이터셋에 대해 동작하는 케라스 모델을 구축하기
def and_sequential():
        data = [[0, 0, 0],
                [0, 1, 0],
                [1, 0, 0],
                [1, 1, 1]]
        data = np.int32(data)  # data를 넘파이로 만들어준다.

        x = data[:, :-1]
        y = data[:, -1:]

        model = keras.Sequential()
        model.add(keras.layers.Dense(1, activation='sigmoid'))

        model.compile(optimizer=keras.optimizers.SGD(0.1),
                      loss=keras.losses.binary_crossentropy,
                      metrics='acc')

        model.fit(x, y, epochs=100, verbose=2)

# AND 데이터셋에 대해 동작하는 케라스 모델(함수형) 구축하기
def and_functional():
        data = [[0, 0, 0],
                [0, 1, 0],
                [1, 0, 0],
                [1, 1, 1]]
        data = np.int32(data)  # data를 넘파이로 만들어준다.

        x = data[:, :-1]
        y = data[:, -1:]

        # 1번
        # inputs = keras.layers.Input([2])        # 함수형으로 쓸 때에는 Inputlayer 대신 Input을 사용한다.
        # dense1 = keras.layers.Dense(2, activation='relu')
        # dense2 = keras.layers.Dense(1, activation='sigmoid')
        #
        # # 이걸 사용하려면 호출해야함. 내부적으로 호출해서 활용. 앞쪽 레이어의 결과를 뒤쪽 레이어가 전달받음
        # output1 = dense1.__call__(inputs)
        # output2 = dense2.__call__(output1)

        # 2번 : 파이썬 문법. __call__생략가능
        # inputs = keras.layers.Input([2])
        # dense1 = keras.layers.Dense(2, activation='relu')
        # dense2 = keras.layers.Dense(1, activation='sigmoid')
        #
        # output1 = dense1(inputs)
        # output2 = dense2(output1)
        #
        # model = keras.Model(inputs, output2)

        # 3번 : 변수를 따로 만들지 않고, 바로 사용하는 방법
        # inputs = keras.layers.Input([2])
        # output1 = keras.layers.Dense(2, activation='relu')(inputs)
        # output2 = keras.layers.Dense(1, activation='sigmoid')(output1)

        # model = keras.Model(inputs, output2)

        # 4번 : 모든 출력의 결과물을 같은 이름으로 한다. 가독성이 좋아짐.
        inputs = keras.layers.Input([2])

        output = keras.layers.Dense(2, activation='relu')(inputs)
        output = keras.layers.Dense(1, activation='sigmoid')(output)

        model = keras.Model(inputs, output)

        # sequencial 결과와 같은 결과가 나온다.
        model.summary()


        # model.compile(optimizer=keras.optimizers.SGD(0.1),
        #               loss=keras.losses.binary_crossentropy,
        #               metrics='acc')
        #
        # model.fit(x, y, epochs=10, verbose=2)

# 2개 이상의 입력을 가지는 함수형 모델 설계
def functional_input():
        data = [[0, 0, 0],
                [0, 1, 0],
                [1, 0, 0],
                [1, 1, 1]]
        data = np.int32(data)

        x1 = data[:, 0:1]
        x2 = data[:, 1:2]
        y  = data[:, 2:3]

        # 멀티 입력을 위해, 입력쪽에 branch 를 만들었다.
        inputs1 = keras.layers.Input([1])
        output1 = keras.layers.Dense(4, activation='relu', name='output1-1')(inputs1)
        output1 = keras.layers.Dense(2, activation='relu', name='output1-2')(output1)

        inputs2 = keras.layers.Input([1])
        output2 = keras.layers.Dense(4, activation='relu', name='output2-1')(inputs2)
        output2 = keras.layers.Dense(2, activation='relu', name='output2-2')(output2)

        # branch 들을 합치는 함수 concatenate
        concat = keras.layers.concatenate([output1, output2])
        output = keras.layers.Dense(1, activation='sigmoid')(concat)

        # 입력이 2개이므로, 입력 2개를 리스트로 묶어주었다.
        model = keras.Model([inputs1, inputs2], output)
        model.summary()

        model.compile(optimizer=keras.optimizers.SGD(0.1),
                      loss=keras.losses.binary_crossentropy,
                      metrics='acc')

        model.fit([x1, x2], y, epochs=10, verbose=2)


# XOR 데이터를 추가하고 출력을 두 개로 확장한 모델 만들기
def functional_input_output():
        data = [[0, 0, 0, 0],
                [0, 1, 0, 1],
                [1, 0, 0, 1],
                [1, 1, 1, 0]]
        data = np.int32(data)

        # 다중 입력과 다중 출력을 위해 x1,x2,y1,y2를 data의 한 열씩 대입.
        x1 = data[:, 0:1]
        x2 = data[:, 1:2]
        y1 = data[:, 2:3]
        y2 = data[:, 3:4]

        inputs1 = keras.layers.Input([1])
        output1 = keras.layers.Dense(4, activation='relu')(inputs1)
        output1 = keras.layers.Dense(2, activation='relu')(output1)

        inputs2 = keras.layers.Input([1])
        output2 = keras.layers.Dense(4, activation='relu')(inputs2)
        output2 = keras.layers.Dense(2, activation='relu')(output2)

        concat = keras.layers.concatenate([output1, output2])

        output3 = keras.layers.Dense(2, activation='relu')(concat)
        output3 = keras.layers.Dense(1, activation='sigmoid')(output3)

        output4 = keras.layers.Dense(2, activation='relu')(concat)
        output4 = keras.layers.Dense(1, activation='sigmoid')(output4)

        model = keras.Model([inputs1, inputs2], [output3, output4])
        model.summary()

        model.compile(optimizer=keras.optimizers.SGD(0.1),
                    loss=keras.losses.binary_crossentropy,
                    metrics='acc')

        model.fit([x1, x2], [y1, y2], epochs=10, verbose=2)


# and_sequential()
# and_functional()
# functional_input()
functional_input_output()