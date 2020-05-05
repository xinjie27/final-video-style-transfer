import cv2

def read_frames(filepath):
    video = cv2.VideoCapture(filepath)
    print("Video successfully opened.")
    count = 0
    is_reading = 1

    while is_reading:
        is_reading, img = video.read()
        cv2.imwrite("frames/fr%d.png" % count, img)
        count += 1