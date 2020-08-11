import numpy as np
from functions import *
from collections import OrderedDict

class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        self.params = {}

        """
        self.params["W1"] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params["b1"] = np.zeros(hidden_size)
        self.params["W2"] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params["b2"] = np.zeros(output_size)

        print("weight : " + str(weight_init_std))
        """
        # He の初期値
        weight_init_std = np.sqrt(2 / input_size)
        self.params["W1"] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params["b1"] = np.zeros(hidden_size)
        weight_init_std = np.sqrt(2 / hidden_size)
        self.params["W2"] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params["b2"] = np.zeros(output_size)

        self.layers = OrderedDict()
        self.layers["Affine1"] = Affine(self.params["W1"], self.params["b1"])
        self.layers["Relu1"] = Relu()
        self.layers["Affine2"] = Affine(self.params["W2"], self.params["b2"])

        self.lastlayer = SoftmaxWithLoss()

        self.correct_num = 0
        self.loss_now = None

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x

    def loss(self, x, t):
        y = self.predict(x)
        yy = softmax(y)
        for di in range(y.shape[0]):
            maxi = 0
            max_x = 0.0
            yes = 0
            for i in range(10):
                if t[di][i] > 0:  yes = i
                if y[di][i] > max_x:
                    max_x = yy[di][i]
                    maxi = i
            if maxi == yes:
                self.correct_num += 1
        return self.lastlayer.forward(y, t)

    def gradient(self, x, t):
        # forward
        self.loss_now = self.loss(x, t)
        
        # backward
        dout = 1
        dout = self.lastlayer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)
        
        grads = {}
        grads["W1"] = self.layers["Affine1"].dW
        grads["b1"] = self.layers["Affine1"].db
        grads["W2"] = self.layers["Affine2"].dW
        grads["b2"] = self.layers["Affine2"].db
        return grads
    
    def learning(self, x_train, t_train):
        iters_num = 100000
        learning_rate = 0.001
        batch_size = min(100, x_train.shape[0])
        
        for i in range(iters_num):
            self.correct_num = 0
            batch_mask = np.random.choice(x_train.shape[0], batch_size)
            x_batch, t_batch = x_train[batch_mask], t_train[batch_mask]
            #x_batch, t_batch = x_train, t_train
            grad = self.gradient(x_batch, t_batch)
            """
            print(self.lastlayer.dx)
            print("test")
            print(np.sum(self.lastlayer.dx))
            """
            for key in ["W1", "b1", "W2", "b2"]:
                self.params[key] -= learning_rate * grad[key]

            if i % (iters_num/50) == 0:
            # if i + 1 == iters_num:
                # print(grad["W1"]*1000)
                # print("are grad[\"W1\"]*1000 ")
                print("loss : " + str(self.loss_now))
                print("grad sum : in iter " + str(i))
                for key in ["W1", "b1", "W2", "b2"]:
                    #print(">> " + key + " : " + str(np.min(grad[key])) + "  " + str(np.sum(grad[key])) + "  " + str(np.max(grad[key])))
                    print(">> " + key + " : " + str(np.min(self.params[key])) + "  " + str(np.sum(self.params[key])) + "  " + str(np.max(self.params[key])))
                print("accuracy num : " + str(self.correct_num) + "   data num : " + str(batch_size))
                print("iter : " + str(i) + " is done.\n")
                
        return
