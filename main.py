import cv2

def detect_shuttlecock(frame):
    # Load the Haar Cascade Classifier for shuttlecock detection
    shuttlecock_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_badminton_shuttlecock.xml')

    # Convert the frame to grayscale for better detection
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect shuttlecocks in the frame
    shuttlecocks = shuttlecock_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Draw rectangles around detected shuttlecocks
    for (x, y, w, h) in shuttlecocks:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    return frame

def main():
    # Open the video file or use a webcam
    video_capture = cv2.VideoCapture('badminton_match.mp4')

    while True:
        # Read a frame from the video
        ret, frame = video_capture.read()

        # Break the loop if the video has ended
        if not ret:
            break

        # Perform shuttlecock detection
        frame_with_detection = detect_shuttlecock(frame)

        # Display the resulting frame
        cv2.imshow('Badminton Match Analysis', frame_with_detection)

        # Break the loop if 'q' key is pressed
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    # Release the video capture object and close the OpenCV windows
    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
