
import os
import numpy as np
import tensornets as nets
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
import random


random.seed(1618)
np.random.seed(1618)
#tf.set_random_seed(1618)   # Uncomment for TF1.
tf.random.set_seed(1618)

#tf.logging.set_verbosity(tf.logging.ERROR)   # Uncomment for TF1.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#ALGORITHM = "guesser"
#ALGORITHM = "tf_net"
ALGORITHM = "tf_conv"

#DATASET = "mnist_d"
#DATASET = "mnist_f"
#DATASET = "cifar_10"
#DATASET = "cifar_100_f"
DATASET = "cifar_100_c"

if DATASET == "mnist_d":
    NUM_CLASSES = 10
    IH = 28
    IW = 28
    IZ = 1
    IS = 784
elif DATASET == "mnist_f":
    NUM_CLASSES = 10
    IH = 28
    IW = 28
    IZ = 1
    IS = 784
elif DATASET == "cifar_10":
    NUM_CLASSES = 10
    IH = 32
    IW = 32
    IZ = 3
    IS = 784
elif DATASET == "cifar_100_f":
    NUM_CLASSES = 100
    IH = 32
    IW = 32
    IZ = 3
    IS = 784                                 # TODO: Add this case.
elif DATASET == "cifar_100_c":
    NUM_CLASSES = 100
    IH = 32
    IW = 32
    IZ = 3
    IS = 784                                 # TODO: Add this case.


#=========================<Classifier Functions>================================

def guesserClassifier(xTest):
    ans = []
    for entry in xTest:
        pred = [0] * NUM_CLASSES
        pred[random.randint(0, 9)] = 1
        ans.append(pred)
    return np.array(ans)


def buildTFNeuralNet(x, y, eps = 6):
    #TODO: Implement a standard ANN here.
    model = keras.Sequential()
    lossType = keras.losses.categorical_crossentropy
    opt = tf.optimizers.Adam()
    inShape=(784,)
    model.add(keras.layers.Dense(NUM_CLASSES, input_shape = inShape, activation=tf.nn.softmax))
    model.compile(optimizer = opt, loss = lossType)
    model.fit(x,y,epochs = 100)
    return model


def buildTFConvNet(x, y, eps = 10, dropout = True, dropRate = 0.2):
    #TODO: Implement a CNN here. dropout option is required.
    model = keras.Sequential()
    inShape = (IH, IW, IZ)
    lossType = keras.losses.categorical_crossentropy
    opt = tf.optimizers.Adam(learning_rate=0.00005)
    if (DATASET == "mnist_d" or DATASET == "mnist_f"):
        model.add(keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=inShape))
        model.add(keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"))
        model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(128, activation="relu"))
        model.add(keras.layers.Dense(NUM_CLASSES, activation="softmax"))
        model.compile(optimizer=opt, loss=lossType,metrics=["accuracy"])
        model.fit(x,y,epochs=200)
    elif DATASET == "cifar_10":
        VGG16_MODEL = tf.keras.applications.VGG19(input_shape=(32, 32, 3),
                                                  include_top=False,
                                                  weights='imagenet')
        global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
        prediction_layer = tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
        model = tf.keras.Sequential([
            VGG16_MODEL,
            global_average_layer,
            prediction_layer
        ])
        opt = tf.optimizers.Adam(learning_rate=0.00005)
        model.compile(optimizer=opt,
                      loss=tf.keras.losses.categorical_crossentropy,
                      metrics=["accuracy"])
        print(x.shape)
        history = model.fit(x, y, epochs=5)
    elif DATASET == "cifar_100_f":
        VGG16_MODEL = tf.keras.applications.VGG19(input_shape=(32, 32, 3),
                                                  include_top=False,
                                                  weights='imagenet')
        global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
        prediction_layer = tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
        model = tf.keras.Sequential([
            VGG16_MODEL,
            global_average_layer,
            prediction_layer
        ])
        opt = tf.optimizers.Adam(learning_rate=0.00001)
        model.compile(optimizer=opt,
                      loss=tf.keras.losses.categorical_crossentropy,
                      metrics=["accuracy"])
        print(x.shape)
        history = model.fit(x, y, epochs=15)
    elif DATASET == "cifar_100_c":
        VGG16_MODEL = tf.keras.applications.VGG19(input_shape=(32, 32, 3),
                                                  include_top=False,
                                                  weights='imagenet')
        global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
        prediction_layer = tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
        model = tf.keras.Sequential([
            VGG16_MODEL,
            global_average_layer,
            prediction_layer
        ])
        opt = tf.optimizers.Adam(learning_rate=0.00001)
        model.compile(optimizer=opt,
                      loss=tf.keras.losses.categorical_crossentropy,
                      metrics=["accuracy"])
        print(x.shape)
        history = model.fit(x, y, epochs=20)
    return model
#=========================<Pipeline Functions>==================================

def getRawData():
    if DATASET == "mnist_d":
        mnist = tf.keras.datasets.mnist
        (xTrain, yTrain), (xTest, yTest) = mnist.load_data()
    elif DATASET == "mnist_f":
        mnist = tf.keras.datasets.fashion_mnist
        (xTrain, yTrain), (xTest, yTest) = mnist.load_data()
    elif DATASET == "cifar_10":
        #pass      # TODO: Add this case.
        (xTrain, yTrain), (xTest, yTest) = tf.keras.datasets.cifar10.load_data()
    elif DATASET == "cifar_100_f":
        #pass      # TODO: Add this case.
        (xTrain, yTrain), (xTest, yTest) = tf.keras.datasets.cifar100.load_data(label_mode="fine")
    elif DATASET == "cifar_100_c":
        #pass      # TODO: Add this case.
        (xTrain, yTrain), (xTest, yTest) = tf.keras.datasets.cifar100.load_data(label_mode="coarse")
    else:
        raise ValueError("Dataset not recognized.")
    print("Dataset: %s" % DATASET)
    print("Shape of xTrain dataset: %s." % str(xTrain.shape))
    print("Shape of yTrain dataset: %s." % str(yTrain.shape))
    print("Shape of xTest dataset: %s." % str(xTest.shape))
    print("Shape of yTest dataset: %s." % str(yTest.shape))
    return ((xTrain, yTrain), (xTest, yTest))



def preprocessData(raw):
    ((xTrain, yTrain), (xTest, yTest)) = raw
    if ALGORITHM != "tf_conv":
        xTrainP = xTrain.reshape((xTrain.shape[0], IS))
        xTestP = xTest.reshape((xTest.shape[0], IS))
    else:
        xTrainP = xTrain.reshape((xTrain.shape[0], IH, IW, IZ))
        xTestP = xTest.reshape((xTest.shape[0], IH, IW, IZ))
    yTrainP = to_categorical(yTrain, NUM_CLASSES)
    yTestP = to_categorical(yTest, NUM_CLASSES)
    print("New shape of xTrain dataset: %s." % str(xTrainP.shape))
    print("New shape of xTest dataset: %s." % str(xTestP.shape))
    print("New shape of yTrain dataset: %s." % str(yTrainP.shape))
    print("New shape of yTest dataset: %s." % str(yTestP.shape))
    return ((xTrainP, yTrainP), (xTestP, yTestP))



def trainModel(data):
    xTrain, yTrain = data
    if ALGORITHM == "guesser":
        return None   # Guesser has no model, as it is just guessing.
    elif ALGORITHM == "tf_net":
        print("Building and training TF_NN.")
        #model = buildTFNeuralNet(xTrain, yTrain)
        return buildTFNeuralNet(xTrain, yTrain)
    elif ALGORITHM == "tf_conv":
        print("Building and training TF_CNN.")
        return buildTFConvNet(xTrain, yTrain)
    else:
        raise ValueError("Algorithm not recognized.")



def runModel(data, model):
    if ALGORITHM == "guesser":
        return guesserClassifier(data)
    elif ALGORITHM == "tf_net":
        print("Testing TF_NN.")
        preds = model.predict(data)
        for i in range(preds.shape[0]):
            oneHot = [0] * NUM_CLASSES
            oneHot[np.argmax(preds[i])] = 1
            preds[i] = oneHot
        return preds
    elif ALGORITHM == "tf_conv":
        print("Testing TF_CNN.")
        preds = model.predict(data)
        for i in range(preds.shape[0]):
            oneHot = [0] * NUM_CLASSES
            oneHot[np.argmax(preds[i])] = 1
            preds[i] = oneHot
        return preds
    else:
        raise ValueError("Algorithm not recognized.")



def evalResults(data, preds):
    xTest, yTest = data
    acc = 0
    for i in range(preds.shape[0]):
        if np.array_equal(preds[i], yTest[i]):   acc = acc + 1
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
