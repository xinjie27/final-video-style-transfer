import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.applications import vgg19, VGG19
from tensorflow.keras.preprocessing.image import load_img, img_to_array, save_img
from tensorflow.compat.v1 import variable_scope, get_variable, Session, global_variables_initializer, train, disable_eager_execution
from tensorflow.keras import backend as K
from tensorflow.keras import optimizers
import argparse
import os
from scipy.optimize import fmin_l_bfgs_b
from vgg19 import VGG
from PIL import ImageOps, Image
from skimage.transform import resize

class Model(object):
    def __init__(self, content, style_filepath, img_h, img_w, lr, frame_idx):
        self.alpha = 1e-4
        self.beta = 1
        self.img_height = img_h
        self.img_width = img_w
        # Layers in which we compute the style loss
        self.style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']
        # Layer in which we compute the content loss
        self.content_layer = 'block5_conv2'
        # Global step and learning rate
        self.gstep = tf.Variable(0, dtype=tf.int32, trainable=False, name="global_step")
        self.lr = lr
        self.frame_idx = frame_idx
        self.opt = optimizers.Adam(self.lr)

        self.gen_input()
        self.load(content, style_filepath)
        self.loss()
    
    def _gen_noise_image(self, content, noise_ratio=0.6):
        noise = np.random.uniform(-20., 20., content.shape).astype(np.float32)
        img = noise_ratio * noise + (1 - noise_ratio) * content
        self.initial_img = img
    
    def _deprocess_img(self, img):
        # img = img.reshape((self.img_height, self.img_width, 3))
        
        # # Inverse mean-centering
        # img[:, :, 0] += 123.68
        # img[:, :, 1] += 116.779
        # img[:, :, 2] += 103.939

        # img = img[:, :, ::-1] # BGR -> RGB
        img = img + self.vgg.mean_pixels
        img = np.clip(img, 0, 255).astype('uint8')
        return img
    
    def load(self, content, style_filepath):
        style = load_img(style_filepath, target_size=(self.img_height, self.img_width))
        style = img_to_array(style)
        self.style = np.expand_dims(style, 0)
        self.content = content
        # content_img = K.variable(self.content)
        # style_img = K.variable(self.style)

        self._gen_noise_image(self.content)
        gen_img = K.variable(self.initial_img)

        # Combine 3 images into a single tensor
        # tensor = K.concatenate([content_img, style_img, gen_img], axis=0)
        # self.model = VGG19(input_tensor=tensor, include_top=False, weights='imagenet')
        # print("VGG19 successfully loaded.")
        # self.layer_outputs = dict([(layer.name, layer.output) for layer in self.model.layers])
        self.vgg = VGG(self.input)
        self.vgg.load()
        self.style -= self.vgg.mean_pixels
        self.content -= self.vgg.mean_pixels

    
    def gen_input(self):
        with variable_scope("input"):
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
        area = img.shape[1] * img.shape[2]
        num_channels = img.shape[3]
        print(img.shape)

        gram_style = self._gram_matrix(style, area, num_channels)
        gram_img = self._gram_matrix(img, area, num_channels)

        loss = tf.math.reduce_sum(tf.math.square(gram_img - gram_style)) / (area * area * num_channels * num_channels * 4)
        return loss

    def _style_loss(self, style_maps):
        """
        Compute the total style loss across all layers

        :param style_maps: a set of all feature maps for the style image
        """
        num_layers = len(style_maps)

        # Initialize the weights for all style layers; this hyperparameter can be tuned
        # General idea: deeper layers are more important
        layer_weights = [0.5 * i + 0.5 for i in range(num_layers)]

        layer_losses = []
        for i in range(num_layers):
            layer_loss = self._layer_style_loss(self.input, style_maps[i]) * layer_weights[i]
            layer_losses.append(layer_loss)

        return sum(layer_losses)

    def loss(self):
        """
        Compute the total loss of the model
        """
        with variable_scope("losses"):
            # Content loss
            with Session() as sess:
                sess.run(self.input.assign(self.content))
                gen_img_content = getattr(self.vgg, self.content_layer)
                content_img_content = sess.run(gen_img_content)
            l_content = self._content_loss(content_img_content, gen_img_content)

            # Style loss
            with Session() as sess:
                sess.run(self.input.assign(self.style))
                style_layers = sess.run([getattr(self.vgg, layer) for layer in self.style_layers])                              
            l_style = self._style_loss(style_layers)
        # # Content loss
        # layer_features = getattr(self.vgg, self.content_layer)
        # print(layer_features.shape)
        # content_out = layer_features[0, :, :, :]
        # combination_out = layer_features[2, :, :, :]

        # l_content = self._content_loss(content_out, combination_out)

        # # Style loss
        # style_maps = []
        # combination_maps = []
        # for layer in self.style_layers:
        #     layer_features = getattr(self.vgg, layer)
        #     style_feature = layer_features[1, :, :, :]
        #     style_maps.append(style_feature)
        #     combination_feature = layer_features[2, :, :, :]
        #     combination_maps.append(combination_feature)
        # style_maps = np.asarray(style_maps)
        # combination_maps = np.asarray(combination_maps)   
        # l_style = self._style_loss(combination_maps, style_maps)

        # Total loss
        self.total_loss = self.alpha * l_content + self.beta * l_style
    
#     def grad(self, img):
#          grads = K.gradients(self.total_loss, img)
#          if len(grads) == 1:
#              grads = grads.flatten().astype('float64')
#          else:
#              grads = np.array(grads).flatten().astype('float64')
#          self.grads = grads

# class Evaluator(object):
#      def __init__(self, model):
#          self.loss_value = None
#          # self.grads_values = None
#          self.model = model

#      def loss(self, img):
#          assert self.loss_value is None
#          self.model.loss(img)
#          self.loss_value = self.model.total_loss
#          return self.model.total_loss

#      def grads(self, img):
#          assert self.loss_value is not None
#          self.model.grad(img)
#          self.loss_value = None
#          # self.grads_values = None
#          return self.model.grads

    def optimize(self):
        self.optimizer = train.AdamOptimizer(self.lr).minimize(self.total_loss, global_step=self.gstep)

    def train(self, n_iters):
        print("Training starts.")
        with Session() as sess:
            sess.run(global_variables_initializer())
            sess.run(self.input.assign(self.initial_img))

            initial_step = self.gstep.eval()

            for epoch in range(initial_step, n_iters):
                print("Current epoch: ", (epoch + 1))
                print("Current loss: ", self.total_loss.eval())
                sess.run(self.optimizer)
                if epoch == (n_iters - 1):
                    gen_img = sess.run([self.input])
                    gen_img = np.asarray(gen_img)
                    final_img = self._deprocess_img(gen_img.copy())
                    filepath = "./frames/frame_%d.png" % self.frame_idx
                    save_img(filepath, final_img)

if __name__ == "__main__":
    os.environ['KMP_DUPLICATE_LIB_OK']='True'
    tf.compat.v1.disable_v2_behavior()

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', required=True, help='path to the input')
    parser.add_argument('-s', '--style_img', required=True, help='path to the style image')
    # parser.add_argument('-o', '--output', required=True, help='output path')
    parser.add_argument('--width', required=False, default=400, type=int, help='output image width')
    parser.add_argument('--height', required=False, default=300, type=int, help='output image height')
    parser.add_argument('--lr', required=False, default=0.001, type=float, help='hyperparameter: learning rate')
    # parser.add_argument('--iter', required=False, default=250, type=int, help='hyperparameter: number of training iterations')

    args = parser.parse_args()
    input_path = args.input
    style_path = args.style_img
    img_height = args.height
    img_width = args.width
    lr = args.lr

    content = load_img(input_path, target_size=(img_height, img_width))
    content = img_to_array(content)
    content = np.expand_dims(content, 0)
    model = Model(content, style_path, img_height, img_width, lr, 0)
    model.optimize()
    model.train(10)