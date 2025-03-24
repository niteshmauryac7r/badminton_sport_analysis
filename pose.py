import torch
import cv2
import warnings
import numpy as np
import pandas as pd


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
    # Add more labels as needed
}

# Set up video writer
fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_video = cv2.VideoWriter('speed&trajectorynew.avi', fourcc, fps, (int(cap.get(3)), int(cap.get(4))))

frame_number = 0

# Initialize dictionaries to store the trajectory of players and shuttlecock
player1_trajectory = {}
player2_trajectory = {}
shuttlecock_trajectory = {}

# Initialize lists to store coordinates and speed for CSV export
coordinates_data = []

# Add this at the beginning of your code to initialize variables for speed analysis
previous_position_player1 = None
previous_position_player2 = None
previous_position_shuttlecock = None
speed_values_player1 = []
speed_values_player2 = []
speed_values_shuttlecock = []

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

            # Track the trajectory and speed of players and shuttlecock
            if cs == 1:  # Player1
                if cs not in player1_trajectory:
                    player1_trajectory[cs] = []
                position = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                player1_trajectory[cs].append(position)

                # Calculate speed if the previous position is available
                if previous_position_player1 is not None:
                    speed_player1 = np.linalg.norm(np.array(position) - np.array(previous_position_player1)) / fps
                    speed_values_player1.append(speed_player1)

                    # Print speed on the video frame
                    cv2.putText(frame, f"Player1 Speed: {speed_player1:.2f} px/s", (10, 90), cv2.FONT_HERSHEY_SIMPLEX,
                                1, (255, 255, 255), 2)

                previous_position_player1 = position

            elif cs == 2:  # Player2
                if cs not in player2_trajectory:
                    player2_trajectory[cs] = []
                position = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                player2_trajectory[cs].append(position)

                # Calculate speed if the previous position is available
                if previous_position_player2 is not None:
                    speed_player2 = np.linalg.norm(np.array(position) - np.array(previous_position_player2)) / fps
                    speed_values_player2.append(speed_player2)

                    # Print speed on the video frame
                    cv2.putText(frame, f"Player2 Speed: {speed_player2:.2f} px/s", (10, 120), cv2.FONT_HERSHEY_SIMPLEX,
                                1, (255, 255, 255), 2)

                previous_position_player2 = position

            elif cs == 0:  # Shuttlecock
                if cs not in shuttlecock_trajectory:
                    shuttlecock_trajectory[cs] = []
                position = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                shuttlecock_trajectory[cs].append(position)

                # Calculate speed if the previous position is available
                if previous_position_shuttlecock is not None:
                    speed_shuttlecock = np.linalg.norm(np.array(position) - np.array(previous_position_shuttlecock)) / fps
                    speed_values_shuttlecock.append(speed_shuttlecock)

                    # Print speed on the video frame
                    cv2.putText(frame, f"Shuttlecock Speed: {speed_shuttlecock:.2f} px/s", (10, 150),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                previous_position_shuttlecock = position

            # Append coordinates and speed to the list for CSV export
            coordinates_data.append({'Frame': frame_number, 'Class': label, 'X': position[0], 'Y': position[1],
                                     'Speed': speed_values_player1[-1] if cs == 1 and speed_values_player1 else
                                              speed_values_player2[-1] if cs == 2 and speed_values_player2 else
                                              speed_values_shuttlecock[-1] if cs == 0 and speed_values_shuttlecock else 0})

        # Draw trajectory lines for players and shuttlecock
        for player, trajectory in [(1, player1_trajectory), (2, player2_trajectory), (0, shuttlecock_trajectory)]:
            if player in trajectory:
                for i in range(1, len(trajectory[player])):
                    cv2.line(frame, trajectory[player][i - 1], trajectory[player][i], class_colors[player], 2)

        # Clear the trajectory data for players and shuttlecock every 100 frames
        if frame_number % 50 == 0:
            player1_trajectory = {}
            player2_trajectory = {}
            shuttlecock_trajectory = {}

        # Write the frame to the output video
        output_video.write(frame)
        print(f"Frame: {frame_number}")
        # Save coordinates data to CSV
        coordinates_df = pd.DataFrame(coordinates_data)
        coordinates_df.to_csv('object_coordinatesnew.csv', index=False)

        # Uncomment the lines below if you want to display the video in a window
        #cv2.namedWindow("video", cv2.WINDOW_NORMAL)
        #cv2.imshow("video", frame)

        if key == ord('q'):
            break
    else:
        continue

#Save coordinates data to CSV
#coordinates_df = pd.DataFrame(coordinates_data)
#coordinates_df.to_csv('object_coordinates.csv', index=False)

cap.release()
output_video.release()
cv2.destroyAllWindows()
