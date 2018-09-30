import cv2
import sys
import math
import numpy as np
import time
from imutils.object_detection import non_max_suppression
from threading import Thread
import threading
import copy


# Constants definition
DRONE_SPEED = 0
global_list = []
group_thresh = 150
frame_rate = 30

# Use HOG Descriptor and Non max suppression to perform person detection
def find_person(image):
    x1 = time.time()
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    rects, weights = hog.detectMultiScale(
        image,
        winStride=(8, 8),
        padding=(8, 8),
        scale=1.05
    )
    rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
    pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)
    ls = [(bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]) for bbox in pick]
    x2 = time.time()
    if x2-x1>0.1:
        print("People detection took a long time of {}".format(x2-x1))
    else:
        print("People detection was quite fast with a time of {}".format(x2-x1))
    return ls

# Create the instance of a tracker
def create_tracker(i = 1):    
    tracker_types = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN']
    tracker_type = tracker_types[i]
    if tracker_type == 'BOOSTING':
        tracker = cv2.TrackerBoosting_create()
    if tracker_type == 'MIL':
        tracker = cv2.TrackerMIL_create()
    if tracker_type == 'KCF':
        tracker = cv2.TrackerKCF_create()
    if tracker_type == 'TLD':
        tracker = cv2.TrackerTLD_create()
    if tracker_type == 'MEDIANFLOW':
        tracker = cv2.TrackerMedianFlow_create()
    if tracker_type == 'GOTURN':
        tracker = cv2.TrackerGOTURN_create()
    return tracker

# Find out if two bbox's overlap beyond a certain threshold
def is_overlap(bbox1, bbox2, min_overlap = 0.7):
    XA1 = bbox1[0]
    XA2 = bbox1[0] + bbox1[2]
    YA1 = bbox1[1]
    YA2 = bbox1[1] + bbox1[3]
    XB1 = bbox2[0]
    XB2 = bbox2[0] + bbox2[2]
    YB1 = bbox2[1]
    YB2 = bbox2[1] + bbox2[3]
    SI = max(0, min(XA2, XB2) - max(XA1, XB1)) * max(0, min(YA2, YB2) - max(YA1, YB1))
    SA = bbox1[2] * bbox1[3]
    SB = bbox2[2] * bbox2[3]
    if SA==0 or SB==0:
        return False
    if SI/SA>min_overlap or SI/SB>min_overlap:
        print("the overlap is {}".format(max(SI/SA, SI/SB)))
        return True
    return False

# Distance between two points
def distance(a, b):
    return ((a[0]-b[0])**2+(a[1]-b[1])**2)**0.5

# Find the group close to the police officer within the threshold
def find_group(list_bbox, police_officer, thresh = group_thresh):
    x = police_officer[0]
    y = police_officer[1]
    ls = []

    list_of_bbox = copy.deepcopy(list_bbox)
    counter = 0

    for i in range(len(list_bbox)):
        if distance(list_bbox[i], police_officer)<thresh:
            ls.append(list_bbox[i])
            list_of_bbox.pop(i-counter)
            counter += 1

    ls1 = []

    for i in range(len(list_of_bbox)):
        for j in range(len(ls)):
            if distance(list_of_bbox[i], ls[j])<thresh:
                ls1.append(list_of_bbox[i])
                break

    ls.extend(ls1)

    min_x, min_y, max_x, max_y = ([],[],[],[])

    for bbox in ls:
        min_x.append(bbox[0])
        min_y.append(bbox[1])
        max_x.append(bbox[0]+bbox[2])
        max_y.append(bbox[1]+bbox[3])

    if min_x!=[]:
        minimum_x = min(min_x)
        minimum_y = min(min_y)
        maximum_x = max(max_x)
        maximum_y = max(max_y)

    return int(minimum_x), int(minimum_y), int(maximum_x), int(maximum_y)

# Find out if the bbox fits the cropping
def is_fit(bbox, threshold):
    x1 = bbox[0]
    y1 = bbox[1]
    x2 = bbox[0]+bbox[2]
    y2 = bbox[1]+bbox[3]
    if x1>threshold[0] and x2<threshold[2] and y1>threshold[1] and y2<threshold[3]:
        return True
    return False

# Class to calculate averages continuously
class Average:
    def __init__(self, items_avg = None):
        self.running_average = 0
        self.number_of_items = 0
        self.items_avg = items_avg
        self.forget_numbers = []

    def moving_average(self, number):
        if self.items_avg:
            if self.number_of_items>self.items_avg:
                self.forget_numbers.pop(0)
            else:
                self.number_of_items += 1
            self.forget_numbers.append(number)
            self.running_average = sum(self.forget_numbers)/len(self.forget_numbers)
        else:
            self.running_average = (self.running_average*self.number_of_items+number)/(self.number_of_items+1)
            self.number_of_items += 1

def clear_up_dic(dic):
    counter = 1
    dic1 = {}
    if dic == {}:
        return {}
    for i in range(max(dic.keys())):
        if i not in dic.keys():
            for key in sorted(dic.keys()):
                if key>i:
                    dic1[key-counter] = dic[key]
            counter += 1
    for i in range(max(dic.keys())-counter+1):
        if i in dic1.keys():
            dic[i] = dic1[i]
    return dic

# Renew the trackers
def renew(minimum_x, maximum_x, minimum_y, maximum_y, prev_min_x, prev_min_y, prev_max_x, prev_max_y, first_time, police_officer, padding, frame, frame_number, final, dic, last_pos, po_tracker, person_found_dic, average_speed_dic):
    global global_list
    global video

    # Crop the frame
    if minimum_x!=None:
        height = frame.shape[0]
        width = frame.shape[1]
        frame= frame[max(0, minimum_y-padding):min(maximum_y+padding, height), max(0, minimum_x-padding):min(maximum_x+padding, width)]
        if first_time:
            first_time = False
            
            #Change the police officer bbox to fit the cropping
            police_officer = (int(police_officer[0]-max(0, minimum_x-padding)), int(police_officer[1]-max(0, minimum_y-padding)), int(police_officer[2]), int(police_officer[3]))
            police_tracker = create_tracker()
            police_tracker.init(frame, police_officer)

            # Change the elements in the 'final' list to fit the cropping
            ls = []
            ppl_count = 0
            dic = {}
            for element in final:
                if not is_fit(element, (minimum_x, minimum_y, maximum_x, maximum_y)):
                    continue
                element = (element[0]-max(0, minimum_x-padding), element[1]-max(0, minimum_y-padding), element[2], element[3])
                ls.append(element)
                dic[ppl_count] = create_tracker()
                dic[ppl_count].init(frame, element)
                ppl_count+=1
            final = copy.deepcopy(ls)

            print("first_time")

        else:
            dif_min_x, dif_min_y = [- prev_min_x + minimum_x, - prev_min_y + minimum_y]

            # Change the bbox of police_officer
            police_officer = (int(police_officer[0]-dif_min_x), int(police_officer[1]-dif_min_y), int(police_officer[2]), int(police_officer[3]))
            police_tracker = create_tracker()
            police_tracker.init(frame, police_officer)


            ls = []
            ppl_count = 0
            dic = {}
            for element in final:
                if not is_fit(element, (minimum_x, minimum_y, maximum_x, maximum_y)):
                    continue
                element = (element[0]-dif_min_x, element[1]-dif_min_y, element[2], element[3])
                ls.append(element)
                dic[ppl_count] = create_tracker()
                dic[ppl_count].init(frame, element)
                ppl_count+=1
            final = copy.deepcopy(ls) 
    
    # Refresh the trackers
    person_bbox = find_person(frame)
    ppl_count = len(dic.values())

    for i in person_found_dic.keys():
        try:
            person_found_dic[i]+=1
        except:
            person_found_dic[i]=1

    #Refresh trackers and see if the police officer can be found again
    for bbox in person_bbox:

        new = True
        # If a bbox overlaps more than 90% with the previous box of the polics officer, it is updated as the new position of the police officer.
        if is_overlap(bbox, police_officer, 0.80):
            continue

        for prev_bbox in final: 
            if is_overlap(bbox, prev_bbox, 0.60):
                x = final.index(prev_bbox)
                final[x] = bbox
                person_found_dic[x] = 0
                new = False
                break

        if new:

            p1 = (int(bbox[0]), int(bbox[1]))
            dic[ppl_count] = create_tracker()
            dic[ppl_count].init(frame, bbox)
            person_found_dic[ppl_count] = 0
            last_pos.append(p1)
            final.append(bbox)
            ppl_count += 1

    person_found_dic1 = copy.deepcopy(person_found_dic)
    # final1 = copy.deepopy(final)
    # last_pos1 = copy.deepcopy(last_pos1)
    # dic1 = copy.deepcopy(dic)
    removing = 1

    for key in person_found_dic.keys():
        if person_found_dic[key]==3:
            del person_found_dic1[key]
            del dic[key]
            average_speed_dic.pop(key-removing)
            final.pop(key-removing)
            last_pos.pop(key-removing)
            removing += 1


    person_found_dic = clear_up_dic(person_found_dic1)
    dic = clear_up_dic(dic)


    if "police_tracker" in locals():
        global_list = [dic, last_pos, first_time, frame_number, final, police_officer, police_tracker, person_found_dic, average_speed_dic]
    else:
        global_list = [dic, last_pos, first_time, frame_number, final, police_officer, po_tracker, person_found_dic, average_speed_dic]

# The main function
def main(video_name=0):
    global global_list
    global video
    global frame_rate
    thread_dic = {}
    average_speed_dic = {}

    
    person_found_dic = {}

    if True:
        # Read video
        video = cv2.VideoCapture(video_name)
        video.set(1, 270)

        # Exit if video not opened.
        if not video.isOpened():
            print("Could not open video")
            sys.exit()

        # Make an object to find the average FPS
        FPS_average = Average()

        done = False

        #Find the  police officer in the people
        while not done:
            # Read first frame.
            ok, frame = video.read()
            if not ok:
                print('Cannot read video file')
                sys.exit()

            # Initialising the trackers
            person_bbox = find_person(frame)
            ppl_count = 0
            dic = {}
            for bbox in person_bbox:
                p1 = (int(bbox[0]), int(bbox[1]))
                p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
                cv2.putText(frame, str(ppl_count), p1, cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
                ppl_count += 1

            #Choose the police officer
            cv2.imshow("initial frame", frame)
            key = cv2.waitKey(0)

            # Press "space" if not all people are recognised immediately. Will skip 10 frames and run people detection again
            if key == 32:
                cv2.destroyWindow("initial frame")
                for i in range(10):
                    ok, frame = video.read()
                continue
            else:
                done = True

            # Store the police officer details and then remove from the generic list of people
            cv2.destroyWindow("initial frame")
            police_officer = person_bbox[int(key)-48]
            person_bbox.pop(int(key)-48)

        remaining_people = 0
        #Create trackers for the remaning people
        for bbox in person_bbox:
            dic[remaining_people] = create_tracker()
            dic[remaining_people].init(frame, bbox)
            remaining_people += 1    

        po_tracker = create_tracker()
        po_tracker.init(frame, police_officer)   

        #constant definitions
        counter = 0
        counter2 = 0
        list_of_bbox = []
        first_time = True
        last_pos = []
        refresh_rate = 5
        padding = 150
        crop_frame = 5
        frame_number = 0
        prev_min_x = 0
        prev_min_y = 0
        prev_max_x = frame.shape[1]
        prev_max_y = frame.shape[0]

    # Start the main loop
    while True:

        # Read a new frame
        ok, frame = video.read()

        if not ok:
            print("video ended")
            sys.exit()

        #update the frame_number
        frame_number += 1

        # Defining what to crop in the frame       
        if counter2==crop_frame:

            # If any person has been detected, then min_x, min_y... should be defined and then the footage can be trimmed.
            if list_of_bbox!=[]:
                if "minimum_x"  in locals():
                    print("changing variables")
                    prev_min_x, prev_min_y, prev_max_x, prev_max_y = [minimum_x, minimum_y, maximum_x, maximum_y]
                minimum_x, minimum_y, maximum_x, maximum_y = find_group(list_of_bbox, police_officer)
                minimum_x, minimum_y, maximum_x, maximum_y = [minimum_x+prev_min_x-min(prev_min_x, padding), minimum_y+prev_min_y-min(prev_min_y, padding), maximum_x+prev_min_x-min(prev_min_x, padding), maximum_y+prev_min_y-min(prev_min_y, padding)]
                counter = refresh_rate
                list_of_bbox = []

            counter2 = 0 

        # Start the thread for refreshing the trackers
        if counter>=refresh_rate:
            print("counter has reached {} frames".format(refresh_rate))
            if "minimum_x" in locals():
                thread_done = False
                print(prev_min_x, prev_min_y, minimum_x, minimum_y)
                thread_dic[counter2] = Thread(target = renew, args = [minimum_x, maximum_x, minimum_y, maximum_y, prev_min_x, prev_min_y, prev_max_x, prev_max_y, first_time, police_officer, padding, frame, frame_number, final, dic, last_pos, po_tracker, person_found_dic, average_speed_dic])
                thread_dic[counter2].start()
            else:
                thread_dic[counter2] = Thread(target = renew, args = [None, None, None, None, prev_min_x, prev_min_y, prev_max_x, prev_max_y, first_time, police_officer, padding, frame, frame_number, final, dic, last_pos, po_tracker, person_found_dic, average_speed_dic])
                thread_dic[counter2].start()
                
            counter = 0

        # If the thread has been updated 
        if global_list!=[]:
            dic, last_pos, first_time, frame_number, final, police_officer, po_tracker, person_found_dic, average_speed_dic = global_list
            global_list = []
            thread_done = True

        # Crop the frame
        if "minimum_x" in locals() and thread_done:
            height = frame.shape[0]
            width = frame.shape[1]
            frame= frame[max(0, minimum_y-padding):min(maximum_y+padding, height), max(0, minimum_x-padding):min(maximum_x+padding, width)]
        else:
            height = frame.shape[0]
            width = frame.shape[1]
            frame= frame[max(0, prev_min_y-padding):min(prev_max_y+padding, height), max(0, prev_min_x-padding):min(prev_max_x+padding, width)]

        # Start timer
        timer = cv2.getTickCount()

        # List which will contain the bounding boxes
        final = []

        # Update tracker
        for tracker in list(dic.values()):
            ok, bbox = tracker.update(frame)
            final.append(bbox)
            if not ok:
                break

        # Update postion of police officer
        ok1, police_officer = po_tracker.update(frame)

        if not ok1:
            print("Police officer not found anymore")
            break

        # Calculate Frames per second (FPS)
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

        # Contains the upper left point of the new positions of the trackers
        new_pos = []

        img = (frame)

        # All people have been tracked successfully
        if ok:
            # Draw the Bounding boxes and the speeds
            for i in range(len(final)):
                
                # Find the speed of the person
                bbox = final[i]
                p1 = (int(bbox[0]), int(bbox[1]))
                p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))

                cv2.putText(img , str(i), p1, cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
                
                if counter2>crop_frame:
                    new_pos.append(p1)
                    if i not in average_speed_dic.keys():
                        average_speed_dic[i] = Average(15)
                    else:
                        try:
                            speed = DRONE_SPEED + np.linalg.norm(np.subtract(p1, last_pos[i]))
                        except:
                            print(last_pos, final)
                            sys.exit()
                        average_speed_dic[i].moving_average(speed)
                        if average_speed_dic[i].running_average>5:
                            cv2.putText(img, "running", p1, cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2) 
                        else:
                            cv2.putText(img, "walking", p1, cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)


                # Show the cropped person
                person = frame[min(p1[1], p2[1]):max(p1[1], p2[1]) , min(p1[0], p2[0]):max(p2[0], p1[0])]
                cv2.imshow(str(i), person)

                proximity = distance(p1, (police_officer[0], police_officer[1]))
                proximity = min(max(255/-95*(proximity-5) + 355, 0), 255)
                cv2.rectangle(img, p1, p2, (0, 255-int(proximity), int(proximity)), 2, 1)

                
            list_of_bbox.extend(final)

            # Update the positions list
            if counter2>crop_frame:
                last_pos = copy.deepcopy(new_pos)

            # Draw the police officer bounding box and label him
            p1 = (int(police_officer[0]), int(police_officer[1]))
            p2 = (int(police_officer[0] + police_officer[2]), int(police_officer[1] + police_officer[3]))
            cv2.rectangle(img, p1, p2, (0, 0, 0), 2, 1)
            cv2.putText(img, "Police Officer", p1, cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2) 
            list_of_bbox.append(police_officer)

        # Tracking failure
        else:
            print("tracking failure")
            counter = refresh_rate
            last_pos = []
            average_speed_dic = {}
            final = []
            list_of_bbox = []
            dic = {}
            continue

        # Add FPS to the average
        FPS_average.moving_average(fps)

        # Display result
        cv2.imshow("Tracking", img)


        # Exit if ESC pressed
        k = cv2.waitKey(1) & 0xff
        if k == 27: break

        # Update the two counters
        counter += 1
        counter2 += 1

        # # Destroy all existing windows
        # cv2.destroyAllWindows()
    print("the average FPS is {}".format(FPS_average.running_average))

if __name__ == '__main__':

    main(0)

