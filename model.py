import tensorflow as tf
from tensorflow.keras.applications import VGG19
from loss_functions import total_loss

def load_vgg19():
    """
    Load the VGG19 network with pre-trained weights.
    """
    model = VGG19(include_top=False, weights='imagenet')
    model.trainable = False
    print("VGG19 model successfully initialized.")
    return model