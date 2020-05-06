import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.applications import VGG19

class Model(object):
    def __init__(self):
        self.learning_rate = 2
        self.alpha = 1e-3
        self.beta = 1
        # Layers in which we compute the style loss
        self.style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']
        # Layer in which we compute the content loss
        self.content_layer = 'block4_conv2'
    
    def load(self):
        self.model = VGG19(include_top=False, weights='imagenet')
        print("VGG19 model successfully loaded.")
        self.layer_outputs = dict([(layer.name, layer.output) for layer in model.layers])

    # This section contains the loss function and four helper functions.
    def _content_loss(self, img, content):
        return tf.math.reduce_sum(tf.math.square(img - content))

    def _gram_matrix(self, img, area, num_channels):
        """
        Compute the gram matrix G for an image tensor

        :param img: the feature map of an image, of shape (h, w, num_channels)
        :param area: h * w for some image
        :param num_channels: the number of channels in some image feature map
        """
        mat = tf.reshape(img, (area, num_channels))
        gram = tf.matmul(tf.transpose(mat), mat)
        return gram

    def _layer_style_loss(self, img, style):
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

    def _style_loss(self, map_set):
        """
        Compute the total style loss across all layers

        :param map_set: a set of all feature maps for the style image
        """
        num_layers = map_set.shape[0]

        # Initialize the weights for all style layers; this hyperparameter can be tuned
        # General idea: deeper layers are more important
        layer_weights = [0.5 * i + 0.5 for i in range(num_layers)]

        layer_losses = []
        for i in range(num_layers):
            layer_loss = _layer_style_loss(map_set[i], self.layer_outputs[self.style_layers[i]]) * layer_weights[i]

        return sum(layer_losses)

    def loss(self, img):
        """
        Compute the total loss of the model
        """
        l_content = self._content_loss(img, content)
        l_style = self._style_loss(img, style)
        self.loss = self.alpha * l_content + self.beta * l_style

    # This section contains image preprocessing and conversion
    # This section trains the model using stochastic gradient descent
    def train(self):
        # TODO
        pass
    