import os
import numpy as np
import tensorflow as tf

from vgg19 import VGG
from utils import *
from os.path import isfile, join
import argparse

from tensorflow.compat.v1 import variable_scope, get_variable, Session, global_variables_initializer, train, disable_eager_execution
tf.compat.v1.disable_eager_execution()

class Image(object):
    def __init__(self, content_filepath, style_filepath, img_h, img_w, lr, frame_idx):
        self.img_height = img_h
        self.img_width = img_w
        
        self.content_img = get_resized_image(content_filepath, img_width, img_height)
        self.style_img = get_resized_image(style_filepath, img_width, img_height)
        self.initial_img = generate_noise_image(self.content_img, img_width, img_height)
        # self.prev_frame = get_resized_image(prev_img, img_width, img_height)
        self.frame_idx = frame_idx

        # Layers in which we compute the content/style loss
        self.content_layer = "block5_conv2"
        self.style_layers = ["block1_conv1", "block2_conv1", "block3_conv1", "block4_conv1", "block5_conv1"]

        # Hyperparameters alpha and beta
        self.content_w = 1e-4
        self.style_w = 1

        # Initialize the weights for all style layers; this hyperparameter can be tuned
        # General idea: deeper layers are more important
        self.style_layer_w = [0.5 * i + 0.5 for i in range(5)]

        self.lr = lr
        self.gstep = tf.Variable(0, dtype=tf.int32, trainable=False, name="global_step")  # global step


    def create_input(self):
        """
        Initialize input tensor
        """
        with tf.compat.v1.variable_scope("input", reuse=tf.AUTO_REUSE):
            self.input_img = tf.compat.v1.get_variable("in_img", shape=([1, self.img_height, self.img_width, 3]), dtype=tf.float32, initializer=tf.zeros_initializer())

    def load_vgg(self):
        self.vgg = VGG(self.input_img)
        self.vgg.load()
        # mean-centering
        self.content_img -= self.vgg.mean_pixels
        self.style_img -= self.vgg.mean_pixels

    # This section contains the loss function and four helper functions.
    def _content_loss(self, img, content):
        self.content_loss = tf.reduce_sum(tf.square(img - content))
        
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

    def _layer_style_loss(self, style, img):
        """
        Compute the style loss in a single layer
        :param img: the input image
        :param style: the style image
        """
        num_channels = style.shape[3]
        area = style.shape[1] * style.shape[2]

        gram_style = self._gram_matrix(style, area, num_channels)
        gram_img = self._gram_matrix(img, area, num_channels)

        return tf.reduce_sum(tf.square(gram_img - gram_style)) / ((2 * area * num_channels) ** 2)

    def _style_loss(self, style_maps):
        """
        Compute the total style loss across all layers

        :param style_maps: a set of all feature maps for the style image
        """
        # We use self.style_layers specified above to compute the total style loss
        num_layers = len(style_maps) # should be 5

        unweighted_loss = [self._layer_style_loss(style_maps[i], getattr(self.vgg, self.style_layers[i]))
             for i in range(num_layers)]

        self.style_loss = sum(self.style_layer_w[i] * unweighted_loss[i] for i in range(num_layers))

    def loss(self):
        """
        Compute the total loss of the model
        """
        with tf.compat.v1.variable_scope("loss", reuse=tf.AUTO_REUSE):
            # Content loss
            with tf.compat.v1.Session() as sess:
                sess.run(self.input_img.assign(self.content_img))
                gen_img_content = getattr(self.vgg, self.content_layer)
                content_img_content = sess.run(gen_img_content)
            self._content_loss(content_img_content, gen_img_content)

            # Style loss
            with tf.compat.v1.Session() as sess:
                sess.run(self.input_img.assign(self.style_img))
                style_layers = sess.run([getattr(self.vgg, layer) for layer in self.style_layers])                              
            self._style_loss(style_layers)

            # Total loss; update to self.total_loss for future optimization
            self.total_loss = self.content_w * self.content_loss + self.style_w * self.style_loss

    def optimize(self):
        self.optimizer = train.AdamOptimizer(self.lr).minimize(self.total_loss, global_step=self.gstep)

    def build(self):
        self.create_input()
        self.load_vgg()
        self.loss()
        self.optimize()
    
    def train(self, n_iters=50):
        with tf.compat.v1.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(self.input_img.assign(self.initial_img))

            initial_step = self.gstep.eval()

            for epoch in range(initial_step, n_iters):
                
                sess.run(self.optimizer)
                if epoch == (n_iters - 1):
                    gen_image, total_loss = sess.run([self.input_img, self.total_loss])
                    # Inverse mean-centering
                    gen_image += self.vgg.mean_pixels 

                    print("Epoch: ", (epoch + 1))
                    print("Loss: ", total_loss)

                    filepath = "./output/frame_%d.png" % self.frame_idx
                    save_image(filepath, gen_image)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', required=True, help='path to the input')
    parser.add_argument('-s', '--style_img', required=True, help='path to the style image')
    # parser.add_argument('-o', '--output', required=True, help='output path')
    parser.add_argument('--width', required=False, default=400, type=int, help='output image width')
    parser.add_argument('--height', required=False, default=300, type=int, help='output image height')
    parser.add_argument('--lr', required=False, default=2., type=float, help='hyperparameter: learning rate')
    # parser.add_argument('--iter', required=False, default=250, type=int, help='hyperparameter: number of training iterations')

    args = parser.parse_args()
    input_path = args.input
    style_path = args.style_img
    img_height = args.height
    img_width = args.width
    lr = args.lr

    model = Image(input_path, style_path, img_height, img_width, lr, 0)
    model.build()
    model.train(10)