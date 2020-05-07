import cv2
import os
import numpy as np
from os.path import isfile, join

def read_frames(filepath):
    video = cv2.VideoCapture(filepath)
    print("Video successfully opened.")
    count = 0  # Frame count
    is_reading = 1

    while is_reading:
        is_reading, img = video.read()
        if (is_reading == False):
            break
        cv2.imwrite("frames/frame_%d.png" % count, img)
        count += 1
    print("%d frames successfully generated." % count)

def frames_to_vid(path_in, path_out, fps):
    frame_array = []
    files = [f for f in os.listdir(path_in) if isfile(join(path_in, f)) if not f.startswith('.')]

    # Sort the file names according to * in frame_*.png
    files.sort(key = lambda x: int(x[6:-4]))

    for i in range(len(files)):
        filename = path_in + files[i]
        # Read each file
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width,height)
        # Insert the frames into an image array
        frame_array.append(img)
    print('Successfully read all files.')

    out = cv2.VideoWriter(path_out, cv2.VideoWriter_fourcc(*'mp4v'), fps, size) # alternatively *'XVID' with .avi

    for i in range(len(frame_array)):
        # Write to a image array
        out.write(frame_array[i])
    out.release()
    print('Output video saved.')


# def main():
#     frames_to_vid('./frames/', 'style_video.mp4', 15) # fps would prob be 30

# if __name__ == "__main__":
#     main()