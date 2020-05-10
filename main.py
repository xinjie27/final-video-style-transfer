import argparse
import numpy as np
import time
from scipy.optimize import fmin_l_bfgs_b
import tensorflow as tf
from tensorflow.keras.preprocessing.image import save_img
from model import Model
from video import *

def main():
    tf.compat.v1.disable_v2_behavior()

    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--video', required=True, help='path to the video')
    parser.add_argument('-s', '--style_img', required=True, help='path to the style image')
    parser.add_argument('-o', '--output', required=True, help='output path')
    parser.add_argument('--fps', required=False, help='frames per second')
    parser.add_argument('--width', required=False, help='output image width')
    parser.add_argument('--height', required=False, help='output image height')
    parser.add_argument('--lr', required=False, help='hyperparameter: learning rate')
    parser.add_argument('--iter', required=False, help='hyperparameter: number of training iterations')

    args = parser.parse_args()
    video_path = args.video
    style_path = args.style_img
    output_path = args.output
    # Parse optional arguments
    fps = args.fps
    img_height = args.height
    img_width = args.width
    lr = args.lr
    n_iters = args.iter

    vid_to_frames(video_path)
    frames_to_vid('./frames/', output_path, fps)

if __name__ == "__main__":
    main()