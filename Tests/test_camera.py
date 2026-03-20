import cv2

def test_camera(device_index=0):
    print(f"Opening camera at device index {device_index}...")
    cap = cv2.VideoCapture(device_index)

    if not cap.isOpened():
        print("ERROR: Could not open camera. Try a different device index (0, 1, 2...)")
        return

    print("Camera opened successfully!")
    print(f"  Resolution: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))} x {int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
    print(f"  FPS: {cap.get(cv2.CAP_PROP_FPS)}")
    print("\nPress 'q' to quit, 's' to save a snapshot.")

    while True:
        ret, frame = cap.read()

        if not ret:
            print("ERROR: Failed to grab frame.")
            break

        cv2.imshow("Camera Test", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Quitting.")
            break
        elif key == ord('s'):
            filename = "snapshot.jpg"
            cv2.imwrite(filename, frame)
            print(f"Snapshot saved as {filename}")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_camera(device_index=0)