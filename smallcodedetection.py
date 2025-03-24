import torch
import cv2
import warnings
import numpy as np

model = torch.hub.load('ultralytics/yolov5', 'custom', 'badminton.pt')

cap = cv2.VideoCapture("matchbadminton.mp4")
sizeStr = str(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))) + 'x' + str(int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define colors for each class
class_colors = {
    0: (255, 0, 0),
    1: (0, 255, 0),
    2: (0, 0, 255)
}

# Define label names for each class
class_labels = {
    0: "Shuttlecock",
    1: "Player1",
    2: "Player2"
}

# Set up video writer
fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_video = cv2.VideoWriter('output_detection.avi', fourcc, fps, (int(cap.get(3)), int(cap.get(4))))

frame_number = 0

while True:
    ret, frame = cap.read()
    if ret:
        detections = model(frame[..., ::-1])
        results = detections.pandas().xyxy[0].to_dict(orient="records")
        key = cv2.waitKey(5)
        warnings.filterwarnings("ignore")

        frame_number += 1

        for result in results:
            con = result['confidence']
            cs = result['class']
            x1 = int(result['xmin'])
            y1 = int(result['ymin'])
            x2 = int(result['xmax'])
            y2 = int(result['ymax'])

            # Get color for the current class
            color = class_colors.get(cs, (0, 0, 0))

            # Get label for the current class
            label = class_labels.get(cs, f"Class {cs}")

            # Display class label along with bounding box in respective color
            label_with_confidence = f"{label} ({con:.2f})"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label_with_confidence, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


        # Write the frame to the output video
        output_video.write(frame)
        print(f"Frame: {frame_number}")

        #cv2.namedWindow("video", cv2.WINDOW_NORMAL)
        #cv2.imshow("video", frame)

        if key == ord('q'):
            break
    else:
        continue

cap.release()
output_video.release()
cv2.destroyAllWindows()
