
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
Layer_NUM = 3
NUM_CLASSES = 10
IMAGE_SIZE = 784

# Use these to set the algorithm to use.
#ALGORITHM = "guesser"
ALGORITHM = "custom_net"
#ALGORITHM = "tf_net"



def kerasANN():
    model = keras.Sequential()
    lossType = keras.losses.categorical_crossentropy
    opt = tf.train.AdamOptimizer()
    inShape=(784,)
    model.add(keras.layers.Dense(NUM_CLASSES, input_shape = inShape, activation=tf.nn.softmax))
    model.compile(optimizer = opt, loss = lossType)
    return model

class NeuralNetwork_2Layer():
    def __init__(self, inputSize, outputSize, neuronsPerLayer, numLayer,learningRate = 0.01):
        self.inputSize = inputSize
        self.outputSize = outputSize
        self.neuronsPerLayer = neuronsPerLayer
        self.lr = learningRate
        self.W1 = np.random.randn(self.inputSize, self.neuronsPerLayer)
        self.W2 = np.random.randn(self.neuronsPerLayer, self.outputSize)
        self.numLayer = numLayer
        self.WList = []
        self.__initializeLayers(self.numLayer)

    # Activation function.
    def __initializeLayers(self, numLayers):
        wList  = []
        wList.append(np.random.randn(self.inputSize, self.neuronsPerLayer))
        for i in range(numLayers -2):
            wList.append(np.random.randn(self.neuronsPerLayer, self.neuronsPerLayer))
        wList.append(np.random.randn(self.neuronsPerLayer, self.outputSize))
        self.W1 = wList[0]
        self.W2 = wList[1]
        self.WList = wList


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
    def train(self, xVals, yVals, epochs = 599, minibatches = True, mbs = 100):
        #pass                                   #TODO: Implement backprop. allow minibatches. mbs should specify the size of each minibatch.
        xBatch = self.__batchGenerator(xVals, mbs)
        yBatch = self.__batchGenerator(yVals, mbs)
        for i in range(epochs):
            xVal = next(xBatch)
            yVal = next(yBatch)
            #lList = self.__forwardmultilayer(xVal)
            l1 , l2 = self.__forward(xVal)
            loss = yVal - l2
            l2Delta = loss * self.__sigmoidDerivative(l2)
            l1Error = np.dot(l2Delta,self.W2.T)
            l1Delta = l1Error * self.__sigmoidDerivative(l1)
            l1Adjust = np.dot(xVal.T, l1Delta) * self.lr
            l2Adjust = np.dot(l1.T,l2Delta) * self.lr

            self.W1 += l1Adjust
            self.W2 += l2Adjust
    def trainmultilayer(self, xVals, yVals, epochs = 599, minibatches = True, mbs = 100):
        #pass                                   #TODO: Implement backprop. allow minibatches. mbs should specify the size of each minibatch.
        xBatch = self.__batchGenerator(xVals, mbs)
        yBatch = self.__batchGenerator(yVals, mbs)
        for k in range(599):
            print(k)
            xVal = next(xBatch)
            yVal = next(yBatch)
            lList = self.__forwardmultilayer(xVal)
            newW = []
            '''
            l1 , l2 = self.__forward(xVal)
            loss = yVal - l2
            l2Delta = loss * self.__sigmoidDerivative(l2)
            l1Error = np.dot(l2Delta,self.W2.T)
            l1Delta = l1Error * self.__sigmoidDerivative(l1)
            l1Adjust = np.dot(xVal.T, l1Delta) * self.lr
            l2Adjust = np.dot(l1.T,l2Delta) * self.lr
            print("l2 Ad")
            print(l2Adjust[1])
            print(l1Error[0][0])
            '''
            loss = yVal - lList[self.numLayer -1]
            Delta = loss * self.__sigmoidDerivative(lList[Layer_NUM -1])
            adjust = np.dot(lList[self.numLayer -2].T, Delta)* self.lr
            #print("Mul l2 Ad")
            #print(adjust[1])
            newW.append(self.WList[self.numLayer-1]+ adjust)
            for j in range(self.numLayer-2,0,-1):
                if (self.numLayer -2 == 0):
                    break
                error = np.dot(Delta,self.WList[j+1].T)
                Delta = error * self.__sigmoidDerivative(lList[j])
                adjust = np.dot(lList[j-1].T,Delta)*self.lr
                newW.append(self.WList[j] + adjust)
            error = np.dot(Delta,self.WList[1].T)
            Delta = error * self.__sigmoidDerivative(lList[0])
            l1Adjust = np.dot(xVal.T,Delta)* self.lr
            newW.append(self.WList[0] + l1Adjust)
            total = len(self.WList)
            for i in range(total):
                self.WList[i] = newW[total - i - 1]


    # Forward pass.
    def __forwardmultilayer(self,input):
        lList = []
        lList.append(self.__sigmoid(np.dot(input, self.WList[0])))
        for i in range(1,self.numLayer):
            lList.append(self.__sigmoid(np.dot(lList[i-1], self.WList[i])))
        return lList

    def __forward(self, input):
        layer1 = self.__sigmoid(np.dot(input, self.W1))
        layer2 = self.__sigmoid(np.dot(layer1, self.W2))
        return layer1, layer2

    # Predict.
    def predict(self, xVals):
        if (self.numLayer < 2):
            _, layer2 = self.__forward(xVals)
            return layer2
        else:
            lList = self.__forwardmultilayer(xVals)
            return lList[self.numLayer-1] 



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
        nn = NeuralNetwork_2Layer(IMAGE_SIZE,NUM_CLASSES,2000,3)
              #TODO: Write code to build and train your custon neural net.
        if (Layer_NUM < 2):
            for i in range(10):
                nn.train(xTrain, yTrain)
        else:
            #nn.__initializeLayers(nn.numLayer)
            for i in range(10):
                nn.trainmultilayer(xTrain, yTrain)
        return nn
    elif ALGORITHM == "tf_net":
        nn = kerasANN()
        nn.fit(xTrain, yTrain,epochs = 100)
        #for x,y in zip(xTrain,yTrain):
         #   print(x.shape)
           # print(y.shape)
            #nn.fit(x, y,epochs = 599)
                         #TODO: Write code to build and train your keras neural net.
        return nn
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
            max = -1
            maxval = 0
            for i in range(10):
                if (result[i] > maxval):
                    max = i
                    maxval = result[i]
            result = np.zeros(10)
            result[max] = 1
            ans.append(result)
        return np.array(ans)
    elif ALGORITHM == "tf_net":
        #print("Testing TF_NN.")
        #print("Not yet implemented.")                  #TODO: Write code to run your keras neural net.
        predicts = model.predict(data)
        ans = []
        for pre in predicts:
            result = pre
            max = -1
            maxval = 0
            for i in range(10):
                if (result[i] > maxval):
                    max = i
                    maxval = result[i]
            result = np.zeros(10)
            result[max] = 1
            ans.append(result)
        return np.array(ans)
    else:
        raise ValueError("Algorithm not recognized.")



def evalResults(data, preds):   #TODO: Add F1 score confusion matrix here.
    xTest, yTest = data
    acc = 0
    f1 = np.zeros((10,10))
    for i in range(preds.shape[0]):
        if np.array_equal(preds[i], yTest[i]):   acc = acc + 1
        tempp = 0
        tempy = 0
        for j in range(10):
            if (preds[i][j] == 1):
                tempp = j
                break
        for j in range(10):
            if (yTest[i][j] == 1):
                tempy = j
                break
        f1[tempp][tempy] +=1
    print("confusion matrix")
    print(f1)
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
