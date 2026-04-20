import cv2
import sys
import numpy as np

PREVIEW   = 0
BLUR      = 1
FEATURES  = 2
CANNY     = 3
BILATERAL = 4
BOX       = 5

feature_params = dict(
    maxCorners=500,
    qualityLevel=0.2,
    minDistance=15,
    blockSize=9
)

def nothing(x):
    pass

def open_device(index):
    cap = cv2.VideoCapture(index)
    if cap.isOpened():
        return cap
    cap.release()
    return None

def scan_devices(max_devices=10):
    devices = []
    for i in range(max_devices):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            devices.append(i)
            cap.release()
    return devices

# scan available cameras
devices = scan_devices(10)
if not devices:
    raise RuntimeError("No camera devices found")

current_idx = 0
device_index = devices[current_idx]
source = open_device(device_index)

image_filter = PREVIEW
alive = True

win_name = "Camera Filters"
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

# Trackbars
# camera
cv2.createTrackbar("Camera", win_name, current_idx, len(devices)-1, nothing)

# bilateral
cv2.createTrackbar("d", win_name, 9, 50, nothing)
cv2.createTrackbar("sigmaColor", win_name, 75, 200, nothing)
cv2.createTrackbar("sigmaSpace", win_name, 75, 200, nothing)

# Box
cv2.createTrackbar("ksize", win_name, 5, 50, nothing)

while alive:
    # handle camera selection
    selected = cv2.getTrackbarPos("Camera", win_name)
    if selected != current_idx:
        new_index = devices[selected]
        new_source = open_device(new_index)
        if new_source:
            source.release()
            source = new_source
            device_index = new_index
            current_idx = selected

    has_frame, frame = source.read()
    if not has_frame:
        break

    frame = cv2.flip(frame, 1)

    if image_filter == PREVIEW:
        result = frame

    elif image_filter == CANNY:
        result = cv2.Canny(frame, 80, 150)

    elif image_filter == BLUR:
        result = cv2.blur(frame, (13, 13))

    elif image_filter == FEATURES:
        result = frame.copy()
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners = cv2.goodFeaturesToTrack(frame_gray, **feature_params)
        if corners is not None:
            pts = np.int32(corners).reshape(-1, 2)
            for x, y in pts:
                cv2.circle(result, (x, y), 10, (0, 255, 0), 1)

    elif image_filter == BILATERAL:
        d = cv2.getTrackbarPos("d", win_name)
        sigmaColor = cv2.getTrackbarPos("sigmaColor", win_name)
        sigmaSpace = cv2.getTrackbarPos("sigmaSpace", win_name)

        if d <= 0:
            d = 1
        if d % 2 == 0:
            d += 1

        result = cv2.bilateralFilter(frame, d, sigmaColor, sigmaSpace)

    elif image_filter == BOX:
        k = cv2.getTrackbarPos("ksize", win_name)
        if k <= 0:
            k = 1
        if k % 2 == 0:
            k += 1
        result = cv2.boxFilter(frame, -1, (k, k))

    cv2.putText(result, f"Camera: {device_index}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow(win_name, result)

    key = cv2.waitKey(1)

    if key in [ord('q'), ord('Q'), 27]:
        alive = False
    elif key in [ord('c'), ord('C')]:
        image_filter = CANNY
    elif key in [ord('b'), ord('B')]:
        image_filter = BLUR
    elif key in [ord('f'), ord('F')]:
        image_filter = FEATURES
    elif key in [ord('p'), ord('P')]:
        image_filter = PREVIEW
    elif key in [ord('o'), ord('O')]:
        image_filter = BILATERAL
    elif key in [ord('i'), ord('I')]:
        image_filter = BOX

source.release()
cv2.destroyAllWindows()