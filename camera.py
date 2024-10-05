import cv2
from pythonProject1 import tester

# def detect_and_display(frame):

def main():
    # Open the default camera
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("Cannot open camera")
        return

    window_name = 'Camera'
    cv2.namedWindow(window_name)

    backSub = cv2.createBackgroundSubtractorMOG2()

    try:
        while True:
            ret, frame = cam.read()
            frame = cv2.flip(frame, 1)
            img_name = "test.png"
            cv2.imwrite(img_name, frame)
            blur1 = cv2.GaussianBlur(frame, (5, 5), 0)
            gray1 = cv2.cvtColor(blur1, cv2.COLOR_BGR2GRAY)
            ret2, thresh1 = cv2.threshold(gray1, 65, 255, cv2.THRESH_BINARY_INV)
            fgMask = backSub.apply(frame)
            cv2.rectangle(frame, (10, 2), (100, 20), (255, 255, 255), -1)
            cv2.putText(frame, str(cam.get(cv2.CAP_PROP_POS_FRAMES)), (15, 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
            # print(tester.main("test.png", "./pythonProject1/waste_classifier_model.pth"))
            if not ret:
                print("Failed to grab frame")
                break

            # Write the frame to the output file
            # out.write(ret)

            # Display the captured frame
            cv2.imshow("filtered", fgMask)
            cv2.imshow("frame", frame)

            # Check if window has been closed
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break

            # Press 'q' to exit the loop
            key = cv2.waitKey(1)
            if key == 27 or key == ord('q') or cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        # Release the capture and writer objects
        cam.release()
        # out.release()
        cv2.destroyAllWindows()
        cv2.waitKey(1)

if __name__ == "__main__":
    main()
