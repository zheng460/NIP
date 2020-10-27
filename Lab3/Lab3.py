
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import backend as K
import random
from PIL import Image
from skimage import transform,io
from scipy.optimize import fmin_l_bfgs_b   # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fmin_l_bfgs_b.html
from tensorflow.keras.applications import vgg19
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from scipy.optimize import fmin_l_bfgs_b
import time
import warnings
random.seed(1618)
np.random.seed(1618)
tf.random.set_seed(1618)
tf.compat.v1.disable_eager_execution()
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

CONTENT_IMG_PATH = r".\content.png"           #TODO: Add this.
STYLE_IMG_PATH = r".\style.png"           #TODO: Add this.

IMAGENET_MEAN_RGB_VALUES = [123.68, 116.779, 103.939]

CONTENT_IMG_H = 500
CONTENT_IMG_W = 500

STYLE_IMG_H = 500
STYLE_IMG_W = 500

CONTENT_WEIGHT = 0.3    # Alpha weight.
STYLE_WEIGHT = 1.0     # Beta weight.
TOTAL_WEIGHT = 1.0

TRANSFER_ROUNDS = 100




#=============================<Helper Fuctions>=================================
'''
TODO: implement this.
This function should take the tensor and re-convert it to an image.
'''


class Evaluator(object):
    def __init__(self,combination_image,loss,grads):
        self.loss_value = None
        self.grads_values = None
        self.loss_gre = fetch_loss_and_grads = K.function([combination_image], [loss, grads])

    def loss(self, x):
        assert self.loss_value is None
        x = x.reshape((1, CONTENT_IMG_H, CONTENT_IMG_W, 3))
        outs = self.loss_gre([x])
        loss_value = outs[0]
        grad_values = outs[1].flatten().astype('float64')
        self.loss_value = loss_value
        self.grad_values = grad_values
        return self.loss_value

    def grads(self, x):
        assert self.loss_value is not None
        grad_values = np.copy(self.grad_values)
        self.loss_value = None
        self.grad_values = None
        return grad_values






#========================<Loss Function Builder Functions>======================









#=========================<Pipeline Functions>==================================

def getRawData():
    print("   Loading images.")
    print("      Content image URL:  \"%s\"." % CONTENT_IMG_PATH)
    print("      Style image URL:    \"%s\"." % STYLE_IMG_PATH)
    cImg = load_img(CONTENT_IMG_PATH, target_size=(CONTENT_IMG_W, CONTENT_IMG_H))
    tImg = cImg.copy()
    sImg = load_img(STYLE_IMG_PATH, target_size=(STYLE_IMG_H, STYLE_IMG_W))
    print("      Images have been loaded.")
    return ((cImg, CONTENT_IMG_H, CONTENT_IMG_W), (sImg, STYLE_IMG_H, STYLE_IMG_W), (tImg, CONTENT_IMG_H, CONTENT_IMG_W))



def preprocessData(raw):
    img, ih, iw = raw
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = vgg19.preprocess_input(img)
    return img


def deprocess_image(x):
    # Remove zero-center by mean pixel
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    # 'BGR'->'RGB'
    x = x[:, :, ::-1]
    x = np.clip(x, 0, 255).astype('uint8')
    return x

'''
TODO: Allot of stuff needs to be implemented in this function.
First, make sure the model is set up properly.
Then construct the loss function (from content and style loss).
Gradient functions will also need to be created, or you can use K.Gradients().
Finally, do the style transfer with gradient descent.
Save the newly generated and deprocessed images.
'''
def content_loss(base, combination):
    return K.sum(K.square(combination - base))

def gram_matrix(x):
    features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
    gram = K.dot(features, K.transpose(features))
    return gram
def style_loss(style, combination):
    S = gram_matrix(style)
    C = gram_matrix(combination)
    channels = 3
    size = STYLE_IMG_H  * STYLE_IMG_W
    return K.sum(K.square(S - C)) / (4. * (channels ** 2) * (size ** 2))

def total_variation_loss(x):
    a = K.square(
        x[:, :CONTENT_IMG_H - 1, :CONTENT_IMG_W - 1, :] - x[:, 1:, :CONTENT_IMG_H - 1, :])
    b = K.square(
        x[:, :STYLE_IMG_W - 1, :STYLE_IMG_H - 1, :] - x[:, :STYLE_IMG_W - 1, 1:, :])
    return K.sum(K.pow(a + b, 1.25))


def style_transfrom(cData,evaluator):
    iterations = 80
    x = cData
    x = x.flatten()
    for i in range(iterations):
        print('Start of iteration', i)
        start_time = time.time()
        x, min_val, info = fmin_l_bfgs_b(evaluator.loss, x,
                                         fprime=evaluator.grads, maxiter=20, maxfun=20)
        print('Current loss value:', min_val)
        # Save current generated image
        img = x.copy().reshape((STYLE_IMG_W, STYLE_IMG_H, 3))
        img = deprocess_image(img)
        im = Image.fromarray(img)
        im.save("result.jpg")
        end_time = time.time()
        print('Iteration %d completed in %ds' % (i, end_time - start_time))

def calculate_loss_function(cData,sData):
    target_image = K.constant(cData)
    style_reference_image = K.constant(sData)
    combination_image = K.placeholder((1, 500, 500, 3))
    input_tensor = K.concatenate([target_image,
                                  style_reference_image,
                                  combination_image], axis=0)
    model = vgg19.VGG19(input_tensor=input_tensor,
                        weights='imagenet',
                        include_top=False)
    # Dict mapping layer names to activation tensors
    outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])
    # Name of layer used for content loss
    content_layer = 'block5_conv2'
    # Name of layers used for style loss
    style_layers = ['block1_conv1',
                    'block2_conv1',
                    'block3_conv1',
                    'block4_conv1',
                    'block5_conv1']
    # Weights in the weighted average of the loss components
    total_variation_weight = 1e-3
    style_weight = 0.5
    content_weight = 0.025
    # Define the loss by adding all components to a `loss` variable
    loss = K.variable(0.)
    layer_features = outputs_dict[content_layer]
    target_image_features = layer_features[0, :, :, :]
    combination_features = layer_features[2, :, :, :]
    loss = loss + content_weight * content_loss(target_image_features,
                                          combination_features)
    for layer_name in style_layers:
        layer_features = outputs_dict[layer_name]
        style_reference_features = layer_features[1, :, :, :]
        combination_features = layer_features[2, :, :, :]
        sl = style_loss(style_reference_features, combination_features)
        loss += (style_weight / len(style_layers)) * sl
    loss += total_variation_weight * total_variation_loss(combination_image)
    return loss, combination_image



#=========================<Main>================================================

def main():
    raw = getRawData()
    cData = preprocessData(raw[0])   # Content image.
    sData = preprocessData(raw[1])   # Style image
    print(" Define loss function")# .
    loss, combination_image = calculate_loss_function(cData,sData)
    print("   Beginning transfer.")
    grads = K.gradients(loss, combination_image)[0]
    evaluator = Evaluator(combination_image,loss,grads)
    style_transfrom(cData,evaluator)
    print("Done.")






if __name__ == "__main__":
    main()