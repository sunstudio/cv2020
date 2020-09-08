import numpy as np
import cv2

def capture_camera():
    cap = cv2.VideoCapture(0)
    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        # Our operations on the frame come here
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Display the resulting frame
        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


def play_video_file():
    src = '../images/vehicle.mp4'
    cap = cv2.VideoCapture(src)
    has_print_prop = False
    while (cap.isOpened()):
        ret, frame = cap.read()
        if not ret: break
        if not has_print_prop:
            has_print_prop = True
            print(cap.get(3), cap.get(4))
            print(frame.shape)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    play_video_file()