This program was developed to count the number of people entering emergency rooms to help hospital management allocate their staff.



https://github.com/JuanFran9/emertech_streamlit/assets/58949950/a8aa4618-4c38-4477-afae-f390257f01cf



It works in the follwing steps:

1. Human head detection using custom trained YOLO computer vision model
2. Tracking (tracking.py)
3. Occupation counting (by taking into account direction of travel of people)

An integration with the slack API was also created to notify the emergency room hospital team.
