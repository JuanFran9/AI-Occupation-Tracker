import cv2
import streamlit as st
import pandas as pd
import numpy as np
from ultralytics import YOLO
from tracker import Tracker
from tracker import *
import matplotlib.pyplot as plt
import requests
import os
import dotenv

dotenv.load_dotenv()

model = YOLO('head_custom_model_v2.pt')


# set up model and tracker
tracker = Tracker()

door_top_left = (700, 80)   #Door top left coordenates
door_bottom_left = (700, 650)  # Door bottom left coordinates
polygon1 = [(door_top_left[0], door_top_left[1] - 10), (door_top_left[0] - 150, door_top_left[1] - 10),
            (door_bottom_left[0] -150, door_bottom_left[1] + 40), (door_bottom_left[0], door_bottom_left[1] + 40)]
polygon2 = [(door_top_left[0] - 170, door_top_left[1] - 10), (door_top_left[0] - 340, door_top_left[1] - 10),
            (door_bottom_left[0] - 340, door_bottom_left[1] + 40), (door_bottom_left[0] - 170, door_bottom_left[1] + 40)]

def send_slack_message(message):
    url = 'https://slack.com/api/chat.postMessage'
    headers = {'Authorization': os.getenv('SLACK_TOKEN')}
    payload = {'channel': '#emergency-room', 'text': message}
    response = requests.post(url, headers=headers, json=payload)
    print(response.text)

def is_inside(polygon, point) -> bool:
    """Returns whether the point is inside the polygon."""
    point = np.array(point, dtype=np.int16)  # convert it in case it's a torch tensor
    dist = cv2.pointPolygonTest(np.array(polygon, np.int32), point, measureDist=False)
    return dist >= 0  # distance is positive when it's inside (and zero when on edge)

def draw_rectangle(frame, cx, cy, w, h, id):
    pc = np.array((cx, cy), dtype=np.int16)
    half_rect = (w / 2, h / 2)
    p1 = (pc - half_rect).astype(np.int16)
    p2 = (pc + half_rect).astype(np.int16)
    cv2.rectangle(frame, p1, p2, (0, 255, 0), 2)
    cv2.circle(frame, pc, 5, (255, 0, 255), -1)
    cv2.putText(frame, str(id), p1, cv2.FONT_HERSHEY_COMPLEX, (5), (255, 255, 255), 2)

avg_occupancy_placeholder = st.empty()

# Function to calculate and display average occupancy
def display_average_occupancy(occupancy_values):
    if occupancy_values:
        average_occupancy = sum(occupancy_values) / len(occupancy_values)
        rounded_avg_occupancy = round(average_occupancy)  # Round to nearest whole number
        # or, to always round down: rounded_avg_occupancy = int(average_occupancy)
        avg_occupancy_placeholder.write(f"Average Occupancy: {rounded_avg_occupancy}")
    else:
        avg_occupancy_placeholder.write("No occupancy data available.")

entering = {}
exiting = {}
entered = []
exited = []
#people_in_left = 0
#people_in_right = 0

# Initialize lists for storing data
occupancy_values = []
frames = []


# Function to update the graph
def update_graph():
    plt.figure(figsize=(10, 4))
    plt.plot(frames, occupancy_values, label='Right', color='blue')
    plt.xlabel('Frames', fontsize = 22)
    plt.ylabel('Occupancy', fontsize = 12)
    plt.title('Occupancy', fontsize = 22)
    plt.legend()
    graph_placeholder.pyplot(plt)


frame_count = 0

# capture video
video_path = "/Users/juanfran/code/JuanFran9/DetechT/raw_data/IMG_3033.MOV"
cap = cv2.VideoCapture(0)
# Streamlit app
st.title("DetechT Video Stream")
# Add a sidebar for the confidence threshold slider
#confidence_threshold = st.sidebar.slider("Select Confidence Threshold", 0.0, 1.0, 0.5)
# Create a Streamlit placeholder for displaying the video frames


# Streamlit app setup
 # Placeholder for the graph

# Add a "Start" button to trigger the execution of the code
if st.sidebar.button("Start"):
    # Create a Streamlit placeholder for displaying the video frames
    output_placeholder = st.empty()
    graph_placeholder = st.empty()

    while True:
        # 1. Extract video frame
        ret, frame = cap.read()
        if not ret:  # no more frames
            break
        #count += 1
        # if count % 3 != 0:
        #     continue
        # Flip the frame to create a mirror image
        frame = cv2.flip(frame, 1)
        #frame = cv2.resize(frame, (1020, 500))  # needed?
        # 2. Get predictions from model
        results = model.predict(source=frame, conf=0.5)
        result = results[0]  # usually there's only one result – we only care for the 1st
        bboxes_frame = result.boxes.xywh
        # 3. Get tracked ids of each box found in frame
        bboxes_tracked = tracker.update(bboxes_frame)
        for bbox_tracked in bboxes_tracked:
            cx, cy, w, h, id = bbox_tracked
            # On entry door, 2 options: entering through that door, or exiting through that door
            if is_inside(polygon1, (cx, cy)):
                draw_rectangle(frame, cx, cy, w, h, id)
                if id in exiting:  # was the person seen in the other door?
                    exited.append(id)
                    del exiting[id]
                else:  # person was not seen before
                    entering[id] = (cx, cy)
            if is_inside(polygon2, (cx, cy)):
                draw_rectangle(frame, cx, cy, w, h, id)
                if id in entering:
                    entered.append(id)
                    del entering[id]
                else:
                    exiting[id] = (cx, cy)

        cv2.polylines(frame,[np.array(polygon2,np.int32)],True,(0,255,0),3) #Define the color eg (255,0,0), and the thickness eg 2
        cv2.putText(frame,str('Entry'),(450,50),cv2.FONT_HERSHEY_COMPLEX,(1.5),(0,255,0),3) #coordinates (504,471) to identify box 1

        cv2.polylines(frame,[np.array(polygon1,np.int32)],True,(0,0,255),3) #Define the color eg (255,0,0), and the thickness eg 2
        cv2.putText(frame,str('Exit'),(650,50),cv2.FONT_HERSHEY_COMPLEX,(1.5),(0,0,255),3)#coordinates (466,485) to identify box 2

        # print(people_entering)
        i = len(entered)
        o = len(exited)
        occupancy = i - o

        # Check if occupancy is above 3 and message has not been sent
        if occupancy > 3 and not message_sent:
            send_slack_message("🚨 ER waitng room above 3 people! 🚨")
        # Set the flag to True to indicate the message has been sent
            message_sent = True
        # Additional code for displaying the message on the frame...

    # If the occupancy drops below 3, reset the message_sent flag
        if occupancy <= 3:
            message_sent = False

        #cv2.putText(frame,str(i),(60,80),cv2.FONT_HERSHEY_COMPLEX,(3),(0,0,255),3)
        #cv2.putText(frame,str(o),(850,80),cv2.FONT_HERSHEY_COMPLEX,(3),(255,0,255),3)

        cv2.rectangle(frame, (780, 5), (1175, 65), (0, 0, 0), -1)
        cv2.putText(frame,f'Occupancy: {occupancy}',(800,50),cv2.FONT_HERSHEY_COMPLEX,(1.5),(255,255,255),3)

        # Display the frame in Streamlit
        output_placeholder.image(frame, channels="BGR", use_column_width=True)
        # Append the current frame count and values to the lists
        frames.append(frame_count)
        occupancy_values.append(occupancy)


        # Update the graph at regular intervals
        if frame_count % 10 == 0:  # Update every 10 frames, adjust as needed
            update_graph()

        # Display the frame in Streamlit
        output_placeholder.image(frame, channels="BGR", use_column_width=True)
        # # Update average occupancy at regular intervals
        # if frame_count % 30 == 0:  # For example, update every 30 frames
        #     display_average_occupancy(occupancy_values)

        frame_count += 1
# Release video capture and close Streamlit app
cap.release()
