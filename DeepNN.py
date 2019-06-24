import numpy as np
from matplotlib import pyplot as plt
'''
#Input
(13, 4) - Input
#4 hidden Layers
3 nodes/neurons - 1st Hidden Layer (Layer 1)
5 nodes/neurons - 2nd Hidden Layer (Layer 2)
4 nodes/neurons - 3rd Hidden Layer (Layer 3)
2 nodes/neurons - 4th Hidden Layer (Layer 4)
#1 Output
1 node - Output (Layer 5)
'''


InputData = np.array([[0, 0, 0, 0],
                      [0, 0, 0, 1], 
                      [0, 1, 0, 0],
                      [0, 1, 0, 1],
                      [0, 1, 1, 0],
                      [0, 1, 1, 1],
                      [1, 0, 0, 0],
                      [1, 0, 0, 1],
                      [1, 0, 1, 0],
                      [1, 0, 1, 1],
                      [1, 1, 0, 0],
                      [1, 1, 0, 1],
                      [1, 1, 1, 1]])

TargetData = np.array([[0], 
                       [1], 
                       [0], 
                       [1], 
                       [1], 
                       [0], 
                       [0], 
                       [1], 
                       [1], 
                       [0], 
                       [0], 
                       [1], 
                       [0]])

TestData = np.array([[1, 1, 1, 0],
                     [0, 0, 1, 0],
                     [0, 0, 1, 1]])


def sigmoid(x):
    return 1 / (1+np.exp(-x))

def sigmoid_p(x):
    return sigmoid(x) * (1-sigmoid(x))


w1 = np.random.randn(4, 4)
b1 = np.random.randn(4, 1)

w2 = np.random.randn(5, 4)
b2 = np.random.randn(5, 1)

w3 = np.random.randn(4, 5)
b3 = np.random.randn(4, 1)

w4 = np.random.randn(2, 4)
b4 = np.random.randn(2, 1)

w5 = np.random.randn(1, 2)
b5 = np.random.randn(1, 1)

iterations = 3000
lr = 0.1
costlist = []

for i in range(iterations):

    z1 = np.dot(w1, InputData.T) + b1
    a1 = sigmoid(z1)

    z2 = np.dot(w2, a1) + b2
    a2 = sigmoid(z2)

    z3 = np.dot(w3, a2) + b3
    a3 = sigmoid(z3)

    z4 = np.dot(w4, a3) + b4
    a4 = sigmoid(z4)
    
    z5 = np.dot(w5, a4) + b5
    a5 = sigmoid(z5)

    cost = np.square(a5 - TargetData.T)
    #print(cost)
    costlist.append(np.sum(cost))

    #backprop

    dcda5 = 2 * (a5 - TargetData.T)
    da5dz5 = sigmoid_p(z5)
    dz5dw5 = a4

    dz5da4 = w5
    da4dz4 = sigmoid_p(z4)
    dz4dw4 = a3

    dz4da3 = w4
    da3dz3 = sigmoid_p(z3)
    dz3dw3 = a2

    dz3da2 = w3
    da2dz2 = sigmoid_p(z2)
    dz2dw2 = a1

    dz2da1 = w2
    da1dz1 = sigmoid_p(z1)
    dz1dw1 = InputData

    dw5 = dcda5 * da5dz5 
    db5 = np.sum(dw5, axis=1, keepdims=True)
    w5 = w5 - lr * np.dot(dw5, dz5dw5.T)
    b5 = b5 - lr * db5

    dw4 = np.dot(dz5da4.T, dw5) * da4dz4
    db4 = np.sum(dw4, axis=1, keepdims=True)
    w4 = w4 - lr * np.dot(dw4, dz4dw4.T)
    b4 = b4 - lr * db4

    dw3 = np.dot(dz4da3.T, dw4) * da3dz3
    db3 = np.sum(dw3, axis=1, keepdims=True)
    w3 = w3 - lr * np.dot(dw3, dz3dw3.T)
    b3 = b3 - lr * db3
    
    dw2 = np.dot(dz3da2.T, dw3) * da2dz2
    db2 = np.sum(dw2, axis=1, keepdims=True)
    w2 = w2 - lr * np.dot(dw2, dz2dw2.T)
    b2 = b2 - lr * db2

    dw1 = np.dot(dz2da1.T, dw2) * da1dz1
    db1 = np.sum(dw1, axis=1, keepdims=True)
    w1 = w1 - lr * np.dot(dw1, dz1dw1)
    b1 = b1 - lr * db1

print("W1 : \n{}\n".format(w1))
print("B1 : \n{}\n".format(b1))

print("W2 : \n{}\n".format(w2))
print("B2 : \n{}\n".format(b2))

print("W3 : \n{}\n".format(w3))
print("B3 : \n{}\n".format(b3))

print("W4 : \n{}\n".format(w4))
print("B4 : \n{}\n".format(b4))

print("W5 : \n{}\n".format(w5))
print("B5 : \n{}\n".format(b5))

z1 = np.dot(w1, InputData.T) + b1
a1 = sigmoid(z1)

z2 = np.dot(w2, a1) + b2
a2 = sigmoid(z2)

z3 = np.dot(w3, a2) + b3
a3 = sigmoid(z3)

z4 = np.dot(w4, a3) + b4
a4 = sigmoid(z4)
    
z5 = np.dot(w5, a4) + b5
a5 = sigmoid(z5)

cost = np.square(a5 - TargetData.T)

print("Prediction : \n{}\n".format(np.round(a5.T)))
print("Cost : \n{}\n".format(np.round(cost.T)))

z1 = np.dot(w1, TestData.T) + b1
a1 = sigmoid(z1)

z2 = np.dot(w2, a1) + b2
a2 = sigmoid(z2)

z3 = np.dot(w3, a2) + b3
a3 = sigmoid(z3)

z4 = np.dot(w4, a3) + b4
a4 = sigmoid(z4)
    
z5 = np.dot(w5, a4) + b5
a5 = sigmoid(z5)

print("Prediction : \n{}\n".format(np.round(a5.T)))


plt.plot(costlist)
plt.show()
