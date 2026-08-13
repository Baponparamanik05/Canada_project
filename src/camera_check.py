import cv2


def main():
    print("Starting camera test...")

    camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if not camera.isOpened():
        print("ERROR: Camera could not be opened.")
        return

    print("Camera opened successfully.")
    print("Press Q to close the camera.")

    while True:
        success, frame = camera.read()

        if not success:
            print("ERROR: Could not read camera frame.")
            break

        cv2.imshow("Camera Test", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()

    print("Camera test completed.")


if __name__ == "__main__":
    main()