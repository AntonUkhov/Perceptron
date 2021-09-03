import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

training_inputs = np.array([[0.5546, 0.6761, 0.5931, 0.9555, 0.1148, 0.8783],
                            [0.5521, 0.6708, 0.5927, 0.9542, 0.1145, 0.8739],
                            [0.5543, 0.6708, 0.5940, 0.9536, 0.1145, 0.8778],
                            [0.5577, 0.6780, 0.5987, 0.9608, 0.1153, 0.8860],
                            [0.5538, 0.6766, 0.5946, 0.9587, 0.1148, 0.8813],
                            [0.5587, 0.6846, 0.5986, 0.9678, 0.1160, 0.8877],
                            [0.5562, 0.6690, 0.5953, 0.9534, 0.1145, 0.8755],
                            [0.5547, 0.6612, 0.5933, 0.9432, 0.1133, 0.8700],
                            [0.5523, 0.6614, 0.5944, 0.9442, 0.1135, 0.8702],
                            [0.5492, 0.6606, 0.5923, 0.9476, 0.1135, 0.8707],
                            [0.5460, 0.6547, 0.5874, 0.9389, 0.1127, 0.8641]])

training_outputs = np.array([[0.7412,
                            0.7405,
                            0.7463,
                            0.7446,
                            0.7519,
                            0.7405,
                            0.7326,
                            0.7335,
                            0.7361,
                            0.7290,
                            0.7272]]).T

np.random.seed(1)

synaptic_weights = 2 * np.random.random((6,1)) - 1

print('случайные инициализирующие веса:')
print(synaptic_weights)


#метод обратного распространения
for i in range(100):
    input_layer = training_inputs
    outputs = sigmoid( np.dot(input_layer, synaptic_weights) )

    err = training_outputs - outputs
    adjustments = np.dot( input_layer.T, err * (outputs * (1 - outputs)) )

    synaptic_weights += adjustments

print('вес после обучения')
print(synaptic_weights)

print('результат после обучения:')
print(outputs)


#тест

new_inputs = np.array([0.5458, 0.6583, 0.5859, 0.9365, 0.1125, 0.8651])
output = sigmoid( np.dot( new_inputs, synaptic_weights) )

print('новая ситуация')
print(output)