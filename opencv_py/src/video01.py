import numpy as np
import cv2

def capture_camera():

    cap = cv2.VideoCapture(0)

    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        # Display the resulting frame
        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


# Python program to illustrate saving an operated video
def camera_save():
    # This will return video from the first webcam on your computer.
    cap = cv2.VideoCapture(0)
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))
    # loop runs if capturing has been initialized.
    while (True):
        # reads frames from a camera
        # ret checks return at each frame
        ret, frame = cap.read()
        if not ret:
            print('cannot read from camera')
            break
        # output the frame
        out.write(frame)
        # The original input frame is shown in the window
        cv2.imshow('preview', frame)
        # Wait for 'a' key to stop the program
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # Close the window / Release webcam
    cap.release()
    # After we release our webcam, we also release the output
    out.release()
    # De-allocate any associated memory usage
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
    # play_video_file()
    camera_save()