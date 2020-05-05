import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications import vgg19, VGG19
from loss_functions import total_loss

def load_img(style_filepath, content_filepath):
    """
    Load the preprocessed style image and content image
    """
    style = plt.imread(style_filepath)
    content = plt.imread(content_filepath)
    # Preprocess the images to satisfy the model requirement
    style = vgg19.preprocess_input(style)
    content = vgg19.preprocess_input(content)
    return style, content


def load_vgg19():
    """
    Load the VGG19 network with pre-trained weights.
    """
    model = VGG19(include_top=False, weights='imagenet')
    model.trainable = False
    print("VGG19 model successfully initialized.")
    return model