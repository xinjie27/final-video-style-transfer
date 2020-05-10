import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.applications import vgg19, VGG19
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.compat.v1 import variable_scope, get_variable, Session, global_variables_initializer, train
from tensorflow.keras import backend as K

class Model(object):
    def __init__(self, content_filepath, style_filepath, img_h=300, img_w=400, lr=2.0):
        self.learning_rate = 2
        self.alpha = 1e-3
        self.beta = 1
        self.img_height = img_h
        self.img_width = img_w
        # Layers in which we compute the style loss
        self.style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']
        # Layer in which we compute the content loss
        self.content_layer = 'block5_conv2'
        # global step and learning rate
        self.gstep = tf.Variable(0, dtype=tf.int32, trainable=False, name="global_step")
        self.lr = lr

        self.gen_input()
        self.load(content_filepath, style_filepath)
    
    def _preprocess_img(self, filepath):
        img = load_img(filepath, target_size=(self.img_height, self.img_width))
        img = img_to_array(img)
        img = np.expand_dims(img, 0)
        img = vgg19.preprocess_input(img)
        return img
    
    def deprocess_img(self, img):
        img = img.reshape((self.img_height, self.img_width, 3))
        img = img[:, :, ::-1] # BGR -> RGB
        img = np.clip(img, 0, 255).astype('uint8')
        # Inverse mean-centering
        img[:, :, 0] += 123.68
        img[:, :, 1] += 116.779
        img[:, :, 2] += 103.939
        return img
    
    def load(self, content_filepath, style_filepath):
        self.content = self._preprocess_img(content_filepath)
        self.style = self._preprocess_img(style_filepath)
        content_img = K.variable(self.content)
        style_img = K.variable(self.style)
        # if K.image_data_format() == 'channels_first':
        #     gen_img = K.placeholder((1, 3, self.img_height, self.img_width))
        # else:
        # gen_img = K.placeholder((1, self.img_height, self.img_width, 3))
        # combine the 3 images into a single Keras tensor
        tensor = K.concatenate([content_img, style_img], axis=0)
        self.model = VGG19(input_tensor=tensor,include_top=False, weights='imagenet')
        print("VGG19 successfully loaded.")
        self.layer_outputs = dict([(layer.name, layer.output) for layer in self.model.layers])
        # Preprocess input images

    
    def gen_input(self):
        with variable_scope("func_gen_input"):
            self.input = get_variable("input", shape=([1, self.img_height, self.img_width, 3]), dtype=tf.float32, initializer=tf.zeros_initializer())

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

        gram_style = self._gram_matrix(style, area, num_channels)
        gram_img = self._gram_matrix(img, area, num_channels)

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
            layer_loss = self._layer_style_loss(map_set[i], self.layer_outputs[self.style_layers[i]]) * layer_weights[i]
            layer_losses.append(layer_loss)

        return sum(layer_losses)

    def loss(self, img):
        """
        Compute the total loss of the model
        """
        with variable_scope("func_loss"):
            # Content loss
            with Session() as sess:
                sess.run(self.input.assign(self.content))
                combination_out = self.layer_outputs[self.content_layer]
                content_out = sess.run(combination_out)
            l_content = self._content_loss(content_out, combination_out)

            # Style loss
            with Session() as sess:
                sess.run(self.input.assign(self.style))
                style_maps = sess.run([self.layer_outputs[layer] for layer in self.style_layers])                 
            l_style = self._style_loss(style_maps)

            # Total loss
            self.total_loss = self.alpha * l_content + self.beta * l_style

    # def grad(self, img):
    #     grads = K.gradients(self.total_loss, img)
    #     if len(grads) == 1:
    #         grads = grads.flatten().astype('float64')
    #     else:
    #         grads = np.array(grads).flatten().astype('float64')
    #     self.grads = grads

    def optimize(self):
        self.optimizer = train.GradientDescentOptimizer(self.lrate).minimize(self.total_loss, global_step=self.gstep)

    def train(self, epochs=250):
        with Session() as sess:
            sess.run(global_variables_initializer())

# class Evaluator(object):
#     def __init__(self, model):
#         self.loss_value = None
#         # self.grads_values = None
#         self.model = model

#     def loss(self, img):
#         assert self.loss_value is None
#         self.model.loss(img)
#         self.loss_value = self.model.total_loss
#         return self.model.total_loss

#     def grads(self, img):
#         assert self.loss_value is not None
#         self.model.grad(img)
#         self.loss_value = None
#         # self.grads_values = None
#         return self.model.grads