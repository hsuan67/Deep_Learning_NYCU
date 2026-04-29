import numpy as np
import matplotlib.pyplot as plt

def generate_linear(n = 100):
    pts = np.random.uniform(0, 1, (n, 2))
    inputs = []
    labels = []
    for pt in pts:
        inputs.append([pt[0], pt[1]])
        distance = (pt[0] - pt[1]) / 1.414
        if pt[0] > pt[1]:
            labels.append(0)
        else:
            labels.append(1)
    return np.array(inputs), np.array(labels).reshape(n, 1)

def generate_XOR_easy():
    inputs = []
    labels = []

    for i in range(11):
        inputs.append([0.1*i, 0.1*i])
        labels.append(0)

        if 0.1*i == 0.5:
            continue

        inputs.append([0.1*i, 1-0.1*i])
        labels.append(1)
    
    return np.array(inputs), np.array(labels).reshape(21, 1)

def show_result(x, y, pred_y, file_name = 'result.png'):
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.title('Ground truth', fontsize = 18)
    for i in range(x.shape[0]):
        if y[i] == 0:
            plt.plot(x[i][0], x[i][1], 'ro')
        else:
            plt.plot(x[i][0], x[i][1], 'bo')
    
    plt.subplot(1, 2, 2)
    plt.title('Predict result', fontsize = 18)
    
    for i in range(x.shape[0]):
        if pred_y[i] < 0.5:
            plt.plot(x[i][0], x[i][1], 'ro')
        else:
            plt.plot(x[i][0], x[i][1], 'bo')

    plt.savefig(file_name)

def show_curve(loss, file_name = 'loss.png'):
    plt.figure()
    plt.title("Learning Curve")
    plt.plot(loss)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.savefig(file_name)

# ----------------------------------------------------

def activation(x, act):
    if act == "Sigmoid":
        return 1.0 / (1.0 + np.exp(-x))
    elif act == "tanh":
        return np.tanh(x)
    elif act == "ReLU":
        return np.maximum(0.0, x)
    elif act == "None":
        return x

def d_activation(x, act):
    if act == "Sigmoid":
        return np.multiply(x, 1.0 - x)
    elif act == "tanh":
        return 1.0 - x ** 2
    elif act == "ReLU":
        x[x <= 0] = 0
        x[x > 0] = 1.0
        return x
    elif act == "None":
        return np.full_like(x, 1.0)

def loss_function(ground_truth, prediction):
    return np.mean((ground_truth - prediction)**2)  # MSE

def d_loss_function(ground_truth, prediction):
    return (-2) * (ground_truth - prediction) / ground_truth.shape[0]

# ----------------------------------------------------

class Layer:
    def __init__(self, input_size, output_size, act):
        # (input_size + 1) x (output_size) 的權重矩陣，每個元素都是從平均值為 0、標準差為 1 的常態分佈中隨機生成數值
        self.weight = np.random.normal(0, 1,(input_size + 1, output_size))  # input size 加一是因為 bias
        self.act = act

    def forward_perLayer(self, input):
        # input.shape[0]: 樣本數
        self.input = np.append(input, np.ones((input.shape[0], 1)), axis=1) # 因為 bias，在 input 添加了一列由 1 組成的向量
        self.output = activation(np.matmul(self.input,self.weight), self.act)
        return self.output

    def backward_perLayer(self, pre_grad):
        self.neu_grad = np.multiply(pre_grad, d_activation(self.output, self.act))  # neu_grad = d_loss/d_output * d_output/d_sum
        return np.matmul(self.neu_grad, self.weight[:-1].T) # neu_grad * all weights(exclude bias)
        
    def update_perLayer(self, learning_rate):
        self.weight_grad = np.matmul(self.input.T, self.neu_grad)   # weight_grad = d_loss/d_weight = neu_grad * d_sum/s_weight
        self.weight -= learning_rate * self.weight_grad

# ----------------------------------------------------

class NeuralNetwork:
    def __init__(self, input_width = 2, hidden_width = 4, output_width = 1, hidden_layers = 2, act = "Sigmoid"):
        self.network = []
        self.network.append(Layer(input_width, hidden_width, act)) # input layer
        
        for i in range(hidden_layers - 1): # hidden layer
            self.network.append(Layer(hidden_width, hidden_width, act))
        
        self.network.append(Layer(hidden_width, output_width, "Sigmoid"))    # output layer

    def forward(self, input):
        for layer in self.network:
            input = layer.forward_perLayer(input)
        return input

    def backward(self, d_loss):
        for layer in reversed(self.network):
            d_loss = layer.backward_perLayer(d_loss)

    def update(self, learning_rate):
        for layer in self.network:
            layer.update_perLayer(learning_rate)

def train(model, feature, label, epoch, learning_rate):
    total_loss = []
    for i in range(epoch):
        model.predict = model.forward(feature)
        loss = loss_function(label, model.predict)
        total_loss.append(loss)
        model.backward(d_loss_function(label, model.predict))
        model.update(learning_rate)

        if (i >= 10000) & (i%5000 == 0):
            print('epoch {:5d} loss : {:.16f}'.format(i, loss))
    return total_loss

def test(model, feature, label):
    predict = model.forward(feature)
    same = 0
    output = []
    for i in range(label.shape[0]):
        
        if label[i,0] > 0.5:
            output.append(1)
        else:
            output.append(0)
        
        if label[i,0] == output[i]:
            same += 1
  
        print("Iter{:2d}  |     Ground truth: {:.1f}  |     prediction: {:.5f}  |     accuracy: {:.8f}  |".format(i, label[i,0], predict[i,0], same/len(label)))
    
    print("loss={:.5f} accuracy={}%".format(loss_function(predict, label), same/len(label)*100 ) )
    
    return output


print("linear:")
# training
linear_model = NeuralNetwork()
linear_x_train, linear_y_train = generate_linear()
linear_train_loss = train(linear_model, linear_x_train, linear_y_train, 100000, 0.1)
show_result(linear_x_train, linear_y_train, linear_model.predict, file_name = 'linear_result.png')
show_curve(linear_train_loss, file_name = 'linear_curve')

# ------------------------Different learning rates------------------------
#plt.figure()
#plt.title("Learning Curve")
#for i in [0.01, 0.05, 0.1, 0.5]:
    #linear_model = NeuralNetwork()
    #linear_train_loss = train(linear_model, linear_x_train, linear_y_train, 100000, i)
    #plt.plot(linear_train_loss, label = 'lr = ' + str(i))
    #print("-"*50)
#plt.legend()
#plt.xlabel("epoch")
#plt.ylabel("loss")
#plt.savefig("linear_lr_loss")

# ------------------------Different numbers of hidden units------------------------
#plt.figure()
#plt.title("Learning Curve")
#for i in [2, 4, 8, 16]:
    #linear_model = NeuralNetwork(i)
    #linear_x_train, linear_y_train = generate_linear()
    #linear_train_loss = train(linear_model, linear_x_train, linear_y_train, 100000, 0.1)
    #plt.plot(linear_train_loss, label = str(i))
    #print("-"*50)
#plt.legend()
#plt.xlabel("epoch")
#plt.ylabel("loss")
#plt.savefig("linear_hidden_loss")

# testing
linear_x_test, linear_y_test = generate_linear()
linear_test = test(linear_model, linear_x_test, linear_y_test, 100000)


print("-----------------------------------------------------------------------------")
print("XOR:")
# training
XOR_model = NeuralNetwork()
XOR_x_train, XOR_y_train = generate_XOR_easy()
XOR_train_loss = train(XOR_model, XOR_x_train, XOR_y_train, 100000, 0.5)
show_result(XOR_x_train,XOR_y_train, XOR_model.predict, file_name = 'XOR_result.png')
show_curve(XOR_train_loss, file_name = 'XOR_curve')

# ------------------------Different learning rates------------------------
#plt.figure()
#plt.title("Learning Curve")
#for i in [0.5, 0.1, 0.05, 0.01]:
    #XOR_model = NeuralNetwork()
    #XOR_train_loss = train(XOR_model, XOR_x_train, XOR_y_train, 100000, i)
    #plt.plot(XOR_train_loss, label = 'lr = ' + str(i))
    #print("-"*50)
#plt.legend()
#plt.xlabel("epoch")
#plt.ylabel("loss")
#plt.savefig("XOR_lr_loss")

# ------------------------Different numbers of hidden units------------------------
#plt.figure()
#plt.title("Learning Curve")
#for i in [2, 4, 8, 16]:
    #XOR_model = NeuralNetwork(i)
    #XOR_x_train, XOR_y_train = generate_linear()
    #XOR_train_loss = train(XOR_model, XOR_x_train, XOR_y_train, 100000, 0.1)
    #plt.plot(XOR_train_loss, label = str(i))
    #print("-"*50)
#plt.legend()
#plt.xlabel("epoch")
#plt.ylabel("loss")
#plt.savefig("XOR_hidden_loss")

# testing
XOR_x_test, XOR_y_test = generate_XOR_easy()
XOR_test = test(XOR_model, XOR_x_test, XOR_y_test, 100000)