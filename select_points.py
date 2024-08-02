import os
import cv2
import numpy as np


convert_to_tuple_of_ints = lambda x: tuple(list(map(int, x)))
VIDEO_PATH = "/home/ubuntu/Downloads/drive-download-20240801T140244Z-001/hammer 1.mp4"
SAVE_FOLDER = "/home/ubuntu/Downloads/drive-download-20240801T140244Z-001/hammer 1"
SCALE = 1

frame_no = 123

image = cv2.imread(os.path.join(SAVE_FOLDER, f"{str(frame_no).zfill(6)}.jpg"))
size = image.shape[1::-1]


WINDOW_NAME = "SELECT"
WINDOWS_SIZE = (int(size[0] * SCALE), int(size[1] * SCALE))
canvas = cv2.resize(image, dsize=WINDOWS_SIZE)

raw_points = np.loadtxt("points.txt").reshape(-1, 2) * SCALE if os.path.isfile("points.txt") else np.array([])


def put_circles(source, points, radius, color=(0, 0, 255), thickness=-1, **kwargs):
    for point in points.astype(int):
        cv2.circle(source, point, radius, color, thickness, **kwargs)


def on_mouse(event, x, y, flags, param):
    global raw_points
    SENSETIVITY_THRESHOLD = 30
    SUPPORTED_EVENTS = (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_RBUTTONDOWN)
    if event in SUPPORTED_EVENTS:
        point = np.array([[x, y]])
        image = np.copy(canvas)

        if event == cv2.EVENT_LBUTTONDOWN:
            raw_points = np.concatenate((raw_points, point), axis=0) if len(raw_points) else point
        if event == cv2.EVENT_RBUTTONDOWN:
            if len(raw_points):
                distances = np.linalg.norm(raw_points - point, axis=-1)

                i = np.argmin(distances)
                if distances[i] < SENSETIVITY_THRESHOLD:
                    raw_points = np.concatenate((raw_points[:i], raw_points[i + 1:]), axis=0)
        put_circles(image, raw_points, radius=3)
        cv2.imshow(WINDOW_NAME, image)


cv2.imshow(WINDOW_NAME, canvas)
cv2.setMouseCallback(WINDOW_NAME, on_mouse)

on_mouse(cv2.EVENT_RBUTTONDOWN, x=-100, y=-100, flags=None, param=None)

while True:
    # cv2.imshow(WINDOW_NAME, canvas)
    k = cv2.waitKey(1) & 0xFF
    if k == ord('c'):
        break
    # elif k == ord('a'):
    #     print mouseX, mouseY

np.savetxt("points.txt", (raw_points.reshape(-1, 2) / SCALE).astype(int), "%i")
arr = np.loadtxt("points.txt") * SCALE
put_circles(canvas, arr, radius=3, color=(255,0,0))
cv2.imshow("A", canvas)
cv2.waitKey()
