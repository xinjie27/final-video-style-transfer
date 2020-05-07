import argparse
import numpy as np
from utils import *
from model import Model, Evaluator

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--video', required=True, help='path to the video')
    parser.add_argument('-o', '--output', required=True, help='output path')
    parser.add_argument('--width', required=False, help='output image width')
    parser.add_argument('--height', required=False, help='output image height')
    parser.add_argument('--iter', required=False, help='hyperparameter: number of training iterations')

    args = parser.parse_args()
    input_path = args.video
    output_path = args.output
    # Parse optional arguments
    img_height = args.height
    img_width = args.width

    read_frames(input_path)

if __name__ == "__main__":
    main()