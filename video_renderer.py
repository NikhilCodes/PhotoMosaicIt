import cv2
from image_mosaicifier import render

camera = cv2.VideoCapture(0)
while True:
    ret, capture = camera.read()
    capture = cv2.resize(capture, (800, 600))
    capture = render(capture)
    cv2.imshow("Mosaicinator", capture)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
