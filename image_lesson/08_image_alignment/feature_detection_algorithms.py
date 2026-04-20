import cv2

# List of detectors
detectors = [
    ("Shi-Tomasi", None),
    ("Harris", None),
    ("FAST", cv2.FastFeatureDetector_create()),
    ("ORB", cv2.ORB_create()),
    ("SIFT", cv2.SIFT_create())
]

index = 0
img_color = cv2.imread("form.jpg")
img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

while True:
    name, detector = detectors[index]
    display = img_color.copy()

    if name == "Shi-Tomasi":
        corners = cv2.goodFeaturesToTrack(img_gray, 200, 0.01, 10)
        if corners is not None:
            for x, y in corners.reshape(-1, 2):
                cv2.circle(display, (int(x), int(y)), 4, (0, 255, 0), -1)

    elif name == "Harris":
        dst = cv2.cornerHarris(img_gray, 2, 3, 0.04)
        dst = cv2.dilate(dst, None)
        display[dst > 0.01 * dst.max()] = [0, 0, 255]

    elif name == "FAST":
        kp = detector.detect(img_gray, None)
        display = cv2.drawKeypoints(display, kp, None, color=(0,255,0))

    elif name == "ORB":
        kp = detector.detect(img_gray, None)
        display = cv2.drawKeypoints(display, kp, None, color=(0,255,0))

    elif name == "SIFT":
        kp = detector.detect(img_gray, None)
        display = cv2.drawKeypoints(display, kp, None, color=(0,255,0))

    cv2.putText(display, name, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("Feature Detectors", display)

    key = cv2.waitKey(0)

    if key in [ord('q'), ord('Q'), 27]:
        break
    elif key in [ord('n'), ord('N')]:
        index = (index + 1) % len(detectors)

cv2.destroyAllWindows()