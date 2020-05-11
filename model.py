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

class Model(object):
    def __init__(self, content, style_filepath, img_h, img_w, lr, frame_idx):
        self.learning_rate = 2
        self.alpha = 1e-3
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

        self.gen_input()
        self.load(content, style_filepath)
        self.loss()
    
    def _gen_noise_image(self, content, noise_ratio=0.6):
        noise = np.random.uniform(-20., 20., content.shape).astype(np.float32)
        img = noise_ratio * noise + (1 - noise_ratio) * content
        self.initial_img = img
    
    def deprocess_img(self, img):
        img = img.reshape((self.img_height, self.img_width, 3))
        
        # Inverse mean-centering
        img[:, :, 0] += 123.68
        img[:, :, 1] += 116.779
        img[:, :, 2] += 103.939

        img = img[:, :, ::-1] # BGR -> RGB
        img = np.clip(img, 0, 255).astype('uint8')
        return img
    
    def load(self, content, style_filepath):
        self.content = vgg19.preprocess_input(content)

        style = load_img(style_filepath, target_size=(self.img_height, self.img_width))
        style = img_to_array(style)
        style = np.expand_dims(style, 0)
        self.style = vgg19.preprocess_input(style)
        content_img = K.variable(self.content)
        style_img = K.variable(self.style)

        self._gen_noise_image(self.content)
        gen_img = K.variable(self.initial_img)

        # combine 3 images into a single tensor
        tensor = K.concatenate([content_img, style_img, gen_img], axis=0)
        self.model = VGG19(input_tensor=tensor,include_top=False, weights='imagenet')
        print("VGG19 successfully loaded.")
        self.layer_outputs = dict([(layer.name, layer.output) for layer in self.model.layers])

    
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
        area = img.shape[0].value * img.shape[1].value
        num_channels = img.shape[2].value

        gram_style = self._gram_matrix(style, area, num_channels)
        gram_img = self._gram_matrix(img, area, num_channels)

        loss = tf.math.reduce_sum(tf.math.square(gram_img - gram_style)) / (area * area * num_channels * num_channels * 4)
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
            style = (self.layer_outputs[self.style_layers[i]])[1,:,:,:]
            layer_loss = self._layer_style_loss(map_set[i], style) * layer_weights[i]
            layer_losses.append(layer_loss)

        return sum(layer_losses)

    def loss(self):
        """
        Compute the total loss of the model
        """
        with variable_scope("losses"):
            # Content loss
            layer_features = self.layer_outputs[self.content_layer]
            content_out = layer_features[0, :, :, :]
            combination_out = layer_features[2, :, :, :]

            l_content = self._content_loss(content_out, combination_out)

            # Style loss
            style_maps = []
            for layer in self.style_layers:
                layer_features = self.layer_outputs[layer]
                style_feature = layer_features[1, :, :, :]
                style_maps.append(style_feature)
            style_maps = np.asarray(style_maps)                 
            l_style = self._style_loss(style_maps)

            # Total loss
            self.total_loss = self.alpha * l_content + self.beta * l_style

    def optimize(self):
        self.optimizer = train.GradientDescentOptimizer(self.lr).minimize(self.total_loss, global_step=self.gstep)

    def train(self, n_iters):
        print("Training starts.")
        with Session() as sess:
            sess.run(global_variables_initializer())
            sess.run(self.input.assign(self.initial_img))

            # (maybe)TODO: train.get_checkpoint_state

            initial_step = self.gstep.eval()

            for epoch in range(initial_step, n_iters):
                print(epoch)
                sess.run(self.optimizer)
                if epoch == (n_iters - 1):
                    gen_img = sess.run([self.input])
                    gen_img = np.asarray(gen_img)
                    gen_img = self.deprocess_img(gen_img)
                    filepath = "./frames/frame_%d.png" % self.frame_idx
                    save_img(filepath, gen_img)

if __name__ == "__main__":
    os.environ['KMP_DUPLICATE_LIB_OK']='True'
    tf.compat.v1.disable_v2_behavior()

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

    content = load_img(input_path, target_size=(img_height, img_width))
    content = img_to_array(content)
    content = np.expand_dims(content, 0)
    model = Model(content, style_path, img_height, img_width, lr, 0)
    model.optimize()
    model.train(300)
    