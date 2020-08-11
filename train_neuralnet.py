import pickle
import numpy as np
import matplotlib.pyplot as plt
from Networks import TwoLayerNet

def look_data(data):
    for k, v in data.items():
        # v = v.reshape(1, v.size)
        print("key : " + k + "     shape : ", end="")
        print(v.shape)
        print(">>   min : " + str(np.min(v)) + "   mean : " + str(np.mean(v)) +  "   max : " + str(np.max(v)) + "  std : " + str(np.std(v)))
        print()
    
class AdaGrad:
    def __init__(self, lr=0.00004):
        self.lr = lr
        self.h = None
    def update(self, params, grads):
        if self.h is None:
            self.h = {}
            for k, v in params.items():
                self.h[k] = np.zeros_like(params[k])

        for key in params.keys():
            self.h[key] += grads[key] * grads[key]
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-7)
            

def main():
    with open(file_path_train, "rb") as f:
        train_data = pickle.load(f)
    x_train, t_train = train_data["x_train"], train_data["t_train"]
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1]*x_train.shape[2]) # flatten
    print("_train.shape")
    print(x_train.shape)
    print(t_train.shape)

    network = TwoLayerNet(x_train.shape[1], 50, 10)

    look_data(network.params)

    iters_num = 10000
    train_size = x_train.shape[0]
    batch_size = 100
    learning_rate = 0.0001

    optimizer = AdaGrad()
    
    train_loss_list = []
    train_acc_list = []
    test_acc_list = []

    iter_per_epoch = max(train_size / batch_size, 1)

    # learning
    for i in range(iters_num):
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]

        grads = network.gradient(x_batch, t_batch)

        """
        for key in ("W1", "b1", "W2", "b2"):
            network.params[key] -= learning_rate * grads[key]
        """
        optimizer.update(network.params, grads)

        if i % (iters_num/20) == 0:
            print("iter : " + str(i) + "\nloss : " + str(network.loss_now))
            # look_data(grads)
        
        train_loss_list.append(network.loss_now)
        
    # Save data
    print("\nSaving neuron params ...")
    with open(file_path_neuron, "wb") as f:
        pickle.dump(network.params, f)
    print("Done! See you.")

    plt.plot(train_loss_list)
    plt.savefig("loss.png")

if __name__ == "__main__":
    file_path_train = "../DATA/train.pkl"
    file_path_neuron = "../DATA/neuron_twolayer.pkl"
    main()

