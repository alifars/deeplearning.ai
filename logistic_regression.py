import numpy as np

x = np.array([[1,2], [3,4]])
y = np.array([[1,0]])


def sigmoid(z):
    s =   1/ (1 + np.exp(-1 * z))
    return s

def initialize_parameters():
    w = np.zeros((2, 1))
    b = np.zeros((1,2))
    return w,b

def y_hat(w, x, b):

    return sigmoid(np.dot(w.T, x) + b)

def calculate_cost(y, y_hat):
    cost = -1/2 * np.sum((np.multiply(y, np.log(y_hat)) + np.multiply((1-y), np.log(1-y_hat))))
    return cost

def backpropgate(x, y, y_hat):

    dw = np.dot(x, (y_hat - y).T)
    db = np.sum(y_hat -y)
    dw /= 2
    db /=2
    grads = dw, db
    return grads

def train():
    # step 1: initialize parameters to zero
    w, b = initialize_parameters()

    epochs = 10000
    learning_rate = 0.01
    for i in range(epochs):
        # in each epoch do the following:
        # 1- propogate forward to calculate y_hat and cost function
        # 2- propogate backward to compute  gradients : dw and db
        # 3- update the weights with graident descent
        y_pred = y_hat(w,x,b)
        cost = calculate_cost(y, y_pred)
        print(cost)
        dw, db = backpropgate(x, y, y_pred)
        w -= learning_rate * dw
        b -= learning_rate * db
    print(w)
    print(b)


train()