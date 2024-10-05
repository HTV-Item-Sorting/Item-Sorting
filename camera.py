import cv2

def main():
    # Open the default camera
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("Cannot open camera")
        return

    # Get the default frame width and height
    frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (frame_width, frame_height))

    window_name = 'Camera'
    cv2.namedWindow(window_name)

    try:
        while True:
            ret, frame = cam.read()
            if not ret:
                print("Failed to grab frame")
                break

            # Write the frame to the output file
            out.write(frame)

            # Display the captured frame
            cv2.imshow(window_name, frame)

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
        out.release()
        cv2.destroyAllWindows()
        cv2.waitKey(1)

if __name__ == "__main__":
    main()
