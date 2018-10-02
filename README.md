# Opencv-Police-Officer-Help
It was used in my friend's project which was to be run on-board a drone that would fly atop the police officer. This code was incorporated with violence detection, but has the following features:
1. Choose a police officer using the keyboard keys and the labels on the bounding boxes.
2. Detect the people in the frame using HOG
3. Finds the group surrouding the police officer 
4. Crops the group so that irrelevant information is left out (i.e., the people who are relatively far from the police officer). The group is continously updated so the crop does too!
5. For each person, according to the distance to the police officer, the color of his/her bounding box will change. It is green when far away, and red when exactly overlapping the police officer. 
6. Using a hardcoded height (i.e., distance from the person to the camera), it can determine whether the person is walking or running.
7. Provides crops of the induvidual people so that weapon analysis can be done on it (weapon analysis is not in this code.)
