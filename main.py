import argparse
import numpy as np
from utils import *
from model import Model, Evaluator
import time
from scipy.optimize import fmin_l_bfgs_b
from tensorflow.keras.preprocessing.image import save_img
import tensorflow as tf

def main():
    tf.compat.v1.disable_v2_behavior()

    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--video', required=True, help='path to the video')
    parser.add_argument('-s', '--style_img', required=True, help='path to the style image')
    parser.add_argument('-o', '--output', required=True, help='output path')
    parser.add_argument('--width', required=False, help='output image width')
    parser.add_argument('--height', required=False, help='output image height')
    parser.add_argument('--iter', required=False, help='hyperparameter: number of training iterations')

    args = parser.parse_args()
    video_path = args.video
    style_path = args.style_img
    output_path = args.output
    # Parse optional arguments
    img_height = args.height
    img_width = args.width
    n_iters = args.iter

    # read_frames(video_path)
    model = Model(video_path, style_path)
    evaluator = Evaluator(model)
    x = get_noise_image(model.content)

    for i in range(300):
        print('Start of iteration', i)
        start_time = time.time()
        x, min_val, info = fmin_l_bfgs_b(evaluator.loss, x.flatten(),
                                        fprime=evaluator.grads, maxfun=20)
        print('Current loss value:', min_val)
        # save current generated image
        img = model.deprocess_img(x.copy())
        fname = '_at_iteration_%d.png' % i
        save_img(fname, img)
        end_time = time.time()
        print('Image saved as', fname)
        print('Iteration %d completed in %ds' % (i, end_time - start_time))

if __name__ == "__main__":
    main()