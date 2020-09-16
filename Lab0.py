
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
import random


# Setting random seeds to keep everything deterministic.
np.set_printoptions(threshold=np.inf)
random.seed(1618)
np.random.seed(1618)
tf.set_random_seed(1618)   # Uncomment for TF1.
#tf.random.set_seed(1618)

# Disable some troublesome logging.
#tf.logging.set_verbosity(tf.logging.ERROR)   # Uncomment for TF1.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Information on dataset.
NUM_CLASSES = 10
IMAGE_SIZE = 784

# Use these to set the algorithm to use.
#ALGORITHM = "guesser"
ALGORITHM = "custom_net"
#ALGORITHM = "tf_net"





class NeuralNetwork_2Layer():
    def __init__(self, inputSize, outputSize, neuronsPerLayer, learningRate = 0.1):
        self.inputSize = inputSize
        self.outputSize = outputSize
        self.neuronsPerLayer = neuronsPerLayer
        self.lr = learningRate
        self.W1 = np.random.randn(self.inputSize, self.neuronsPerLayer)
        self.W2 = np.random.randn(self.neuronsPerLayer, self.outputSize)

    # Activation function.
    def __sigmoid(self, x):
        #pass   #TODO: implement
        return 1 / (1 + np.exp(-x))

    # Activation prime function.
    def __sigmoidDerivative(self, x):
        #pass   #TODO: implement
        return (self.__sigmoid(x))* (1-self.__sigmoid(x))

    # Batch generator for mini-batches. Not randomized.
    def __batchGenerator(self, l, n):
        for i in range(0, len(l), n):
            yield l[i : i + n]

    # Training with backpropagation.
    def train(self, xVals, yVals, epochs = 500, minibatches = True, mbs = 100):
        #pass                                   #TODO: Implement backprop. allow minibatches. mbs should specify the size of each minibatch.
        xBatch = self.__batchGenerator(xVals, mbs)
        yBatch = self.__batchGenerator(yVals, mbs)
        for i in range(epochs):
            xVal = next(xBatch)
            yVal = next(yBatch)

            l1 , l2 = self.__forward(xVal)
            loss = yVal - l2
            l2Delta = loss * self.__sigmoidDerivative(l2)
            l1Error = np.dot(l2Delta,self.W2.T)
            l1Delta = l1Error * self.__sigmoidDerivative(l1)
            l1Adjust = np.dot(xVal.T, l1Delta) * self.lr
            l2Adjust = np.dot(l1.T,l2Delta) * self.lr

            self.W1 = self.W1 + l1Adjust
            self.W2 = self.W2 + l2Adjust
            print(i)
        print(self.W1.shape)
        print(self.W2.shape)
    # Forward pass.
    def __forward(self, input):
        layer1 = self.__sigmoid(np.dot(input, self.W1))
        layer2 = self.__sigmoid(np.dot(layer1, self.W2))
        return layer1, layer2

    # Predict.
    def predict(self, xVals):
        _, layer2 = self.__forward(xVals)
        return layer2



# Classifier that just guesses the class label.
def guesserClassifier(xTest):
    ans = []
    for entry in xTest:
        pred = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        pred[random.randint(0, 9)] = 1
        ans.append(pred)
    return np.array(ans)

#def SigClassifier(xTest,nn):
    #passl1 , l2 = nn.__forward(xTest)



#=========================<Pipeline Functions>==================================

def getRawData():
    mnist = tf.keras.datasets.mnist
    (xTrain, yTrain), (xTest, yTest) = mnist.load_data()
    print("Shape of xTrain dataset: %s." % str(xTrain.shape))
    print("Shape of yTrain dataset: %s." % str(yTrain.shape))
    print("Shape of xTest dataset: %s." % str(xTest.shape))
    print("Shape of yTest dataset: %s." % str(yTest.shape))
    #print(xTrain[0])
    return ((xTrain, yTrain), (xTest, yTest))



def preprocessData(raw):
    ((xTrain, yTrain), (xTest, yTest)) = raw          #TODO: Add range reduction here (0-255 ==> 0.0-1.0).
    yTrainP = to_categorical(yTrain, NUM_CLASSES)
    yTestP = to_categorical(yTest, NUM_CLASSES)
    xTrain = xTrain.reshape(60000,784)
    xTest = xTest.reshape(10000,784)
    xTrain = xTrain/256.0
    xTest = xTest/256.0

    print("New shape of xTrain dataset: %s." % str(xTrain.shape))
    print("New shape of xTest dataset: %s." % str(xTest.shape))
    print("New shape of yTrain dataset: %s." % str(yTrainP.shape))
    print("New shape of yTest dataset: %s." % str(yTestP.shape))
    return ((xTrain, yTrainP), (xTest, yTestP))



def trainModel(data):
    xTrain, yTrain = data
    if ALGORITHM == "guesser":
        return None   # Guesser has no model, as it is just guessing.
    elif ALGORITHM == "custom_net":
        #print("Building and training Custom_NN.")
        nn = NeuralNetwork_2Layer(IMAGE_SIZE,NUM_CLASSES,IMAGE_SIZE)
        nn.train(xTrain, yTrain)               #TODO: Write code to build and train your custon neural net.
        return nn
    elif ALGORITHM == "tf_net":
        print("Building and training TF_NN.")
        print("Not yet implemented.")                   #TODO: Write code to build and train your keras neural net.
        return None
    else:
        raise ValueError("Algorithm not recognized.")



def runModel(data, model):
    if ALGORITHM == "guesser":
        return guesserClassifier(data)
    elif ALGORITHM == "custom_net":
        print("Testing Custom_NN.")
        #print("Not yet implemented.")                   #TODO: Write code to run your custon neural net.
        #return None
        ans = []
        for entry in data:
            result = model.predict(entry)
            for i in range(10):
                if (result[i] > 0.5):
                    result[i] = 1
                else :
                    result[i] = 0
            ans.append(result)
        return np.array(ans)
    elif ALGORITHM == "tf_net":
        print("Testing TF_NN.")
        print("Not yet implemented.")                   #TODO: Write code to run your keras neural net.
        return None
    else:
        raise ValueError("Algorithm not recognized.")



def evalResults(data, preds):   #TODO: Add F1 score confusion matrix here.
    xTest, yTest = data
    acc = 0
    for i in range(preds.shape[0]):
        if np.array_equal(preds[i], yTest[i]):   acc = acc + 1
    print(preds[0],yTest[0])
    accuracy = acc / preds.shape[0]
    print("Classifier algorithm: %s" % ALGORITHM)
    print("Classifier accuracy: %f%%" % (accuracy * 100))
    print()



#=========================<Main>================================================

def main():
    raw = getRawData()
    data = preprocessData(raw)
    model = trainModel(data[0])
    preds = runModel(data[1][0], model)
    evalResults(data[1], preds)



if __name__ == '__main__':
    main()
