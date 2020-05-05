import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications import vgg19, VGG19
from loss_functions import loss

class Model():
    def __init__(self):
        super().__init__()
    
    # Load the VGG19 network with pre-trained weights.
    model = VGG19(include_top=False, weights='imagenet')
    model.trainable = False
    print("VGG19 model successfully loaded.")

    # LOSS: This section contains the loss function and four helper functions.
    def _content_loss(img, content):
        return tf.math.reduce_sum(tf.math.square(img - content))

    def _gram_matrix(img, area, num_channels):
        """
        Compute the gram matrix G for an image tensor

        :param img: the feature map of an image, of shape (h, w, num_channels)
        :param area: h * w for some image
        :param num_channels: the number of channels in some image feature map
        """
        mat = tf.reshape(img, (area, num_channels))
        gram = tf.matmul(tf.transpose(mat), mat)
        return gram

    def _layer_style_loss(img, style):
        """
        Compute the style loss in a single layer

        :param img: the input image
        :param style: the style image
        """
        h, w, num_channels = img.shape
        area = h * w

        gram_style = _gram_matrix(style, area, num_channels)
        gram_img = _gram_matrix(img, area, num_channels)

        loss = tf.math.reduce_sum(tf.math.square(gram_img - gram_style)) / (area * num_channels * 2)**2
        return loss

    def _style_loss(map_set):
        """
        Compute the total style loss across all layers

        :param map_set: a set of all feature maps of the style image
        """
        num_layers = map_set.shape[0]

        # Initialize the weights for all style layers; this hyperparameter can be tuned
        # General idea: deeper layers are more important
        layer_weights = [0.5 * i + 0.5 for i in range(num_layers)]

        layer_losses = []
        for i in range(num_layers):
            layer_loss = _layer_style_loss(map_set[i], ) * layer_weights[i]

        return sum(layer_losses)

    def loss(img, content, style, alpha=10, beta=80):
        l_content = _content_loss(img, content)
        l_style = _style_loss(img, style)
        loss = alpha * l_content + beta * l_style
        return loss