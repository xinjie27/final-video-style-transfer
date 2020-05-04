import cv2

def read_frames(filepath):
    video = cv2.VideoCapture(filepath)
    count = 0
    is_reading = 1

    while is_reading:
        is_reading, img = video.read()
        cv2.imwrite("frame_%d.jpg" % count, img)
        count += 1

    pass