from threading import Thread
import cv2
import mediapipe as mp
import numpy as np
import csv
from csv import reader
import pyttsx3

def squat(c):
    cap = cv2.VideoCapture(0)

    mp_drawing=mp.solutions.drawing_utils
    mp_pose=mp.solutions.pose

    fieldnames = ["counter"]
    counter = 0
    position = None
    msg = None

    cap = cv2.VideoCapture(0)

    def calculate_angle(a,b,c):
        a = np.array(a) 
        b = np.array(b) 
        c = np.array(c) 
        
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians*180.0/np.pi)
        
        if angle >180.0:
            angle = 360-angle
            
        return angle

    # Csv file setup
    with open('data.csv', 'w') as csv_file:
        csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        csv_writer.writeheader()

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while True:
            with open('data.csv', 'a') as csv_file:
                csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                ret, frame = cap.read()
            
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
            
                results = pose.process(image)
            
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                right_hip=[0,0]
                try:
                
                    landmarks = results.pose_landmarks.landmark
                    left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                    right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                    right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                    left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                    left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                    right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]

                    left_hip_angle = calculate_angle(left_shoulder,left_hip,left_knee)
                    right_hip_angle = calculate_angle(right_shoulder,right_hip,right_knee)

                    cv2.putText(image,str(right_hip_angle), 
                                        tuple(np.multiply(left_hip, [640, 480]).astype(int)), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)        


                    if left_hip_angle > 170 and right_hip_angle > 170:
                        position='UP'
                        msg='Correct'
                    if left_hip_angle < 100 and right_hip_angle < 100 and position =='UP':
                        if left_hip_angle and right_hip_angle >= 90:
                            counter +=1                
                            msg='ERROR'
                            position="DOWN"
                            info = {"counter": msg}
                            csv_writer.writerow(info)
                        else:
                            msg='Correct'
                            position="DOWN"
                            info = {"counter": counter}
                            csv_writer.writerow(info)                            
                            position=None
                            msg=None
                    elif left_hip_angle and right_hip_angle <= 90:
                        msg='ERROR'
                        position="DOWN"
                        print(right_hip_angle)
                        print(msg)
                        info = {"counter": msg}
                        csv_writer.writerow(info)
                except:
                    pass

                cv2.rectangle(image, (0,0), (70,70), (250,228,13), -1)
                cv2.putText(image, 'REPS', (15,12), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 1, cv2.LINE_AA)        
                cv2.putText(image, str(counter), 
                            (10,60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)        
                # cv2.putText(image, str(position), 
                #             tuple(np.multiply(right_hip, [640, 480]).astype(int)), 
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)        
                cv2.rectangle(image, (550,0), (640,60), (250,228,13), -1)
                cv2.putText(image,str(msg), 
                        (550,40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2, cv2.LINE_AA)
                if msg=='Correct':
                    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                            mp_drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=2), 
                                            mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2) 
                                            )               
                elif msg=='ERROR':
                    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                            mp_drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=2), 
                                            mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2) 
                                            )               
                cv2.imshow('Live Feed', image)

                if counter==c or cv2.waitKey(10) & 0xFF == ord('q'):
                    break
        cap.release()
        cv2.destroyAllWindows()
        return True
def lunges(c):
    cap = cv2.VideoCapture(0)

    mp_drawing=mp.solutions.drawing_utils
    mp_pose=mp.solutions.pose
    fieldnames = ["counter"]
    counter = 0     

    position = None
    msg = None

    def calculate_angle(a,b,c):
        a = np.array(a) 
        b = np.array(b) 
        c = np.array(c) 
        
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians*180.0/np.pi)
        
        if angle >180.0:
            angle = 360-angle
            
        return angle

    # Csv file setup
    with open('data.csv', 'w') as csv_file:
        csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        csv_writer.writeheader()

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while True:
            with open('data.csv', 'a') as csv_file:
                csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                ret, frame = cap.read()
            
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
            
                results = pose.process(image)
            
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                right_hip=[0,0]
                try:
                
                    landmarks = results.pose_landmarks.landmark
                    right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                    left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                    left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                    right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                    left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                    right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

                    left_leg = calculate_angle(left_hip,left_knee,left_ankle)
                    right_leg = calculate_angle(right_hip,right_knee,right_ankle)

                    cv2.putText(image,str(right_leg), 
                                        tuple(np.multiply(left_hip, [640, 480]).astype(int)), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)        


                    if left_leg > 170 or right_leg > 170:
                        position='UP'
                        msg='Correct'
                    if (left_leg < 90 or right_leg < 100) and position =='UP':
                        if left_leg and right_leg <= 90:
                             msg='ERROR'
                             position="DOWN"
                            #  print('right:',right_leg,'left:',left_leg)
                            #  print(msg)
                             info = {"counter": msg}
                             csv_writer.writerow(info)
                        elif left_leg and right_leg >= 90:
                            msg='Correct'
                            position="DOWN"
                            counter +=1
                            info = {"counter": counter}
                            csv_writer.writerow(info)
                            position=None
                            msg=None
                except:
                    pass

                cv2.rectangle(image, (0,0), (70,70), (250,228,13), -1)
                cv2.putText(image, 'REPS', (15,12), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 1, cv2.LINE_AA)        
                cv2.putText(image, str(counter), 
                            (10,60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)        
                # cv2.putText(image, str(position), 
                #             tuple(np.multiply(right_hip, [640, 480]).astype(int)), 
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)        
                cv2.rectangle(image, (550,0), (640,60), (250,228,13), -1)
                cv2.putText(image,str(msg), 
                        (550,40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2, cv2.LINE_AA)
                if msg=='Correct':
                    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                            mp_drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=2), 
                                            mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2) 
                                            )               
                elif msg=='ERROR':
                    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                            mp_drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=2), 
                                            mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2) 
                                            )               
                cv2.imshow('Live Feed', image)

                if counter==c or cv2.waitKey(10) & 0xFF == ord('q'):
                    break

        cap.release()
        cv2.destroyAllWindows()
def highknees():
    mp_drawing=mp.solutions.drawing_utils
    mp_pose=mp.solutions.pose

    fieldnames = ["counter"]
    counter = 0     

    position = None
    msg = None

    cap = cv2.VideoCapture(0)

    def calculate_angle(a,b,c):
        a = np.array(a) 
        b = np.array(b) 
        c = np.array(c) 
        
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians*180.0/np.pi)
        
        if angle >180.0:
            angle = 360-angle
            
        return angle

    # Csv file setup
    with open('data.csv', 'w') as csv_file:
        csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        csv_writer.writeheader()

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while True:
            with open('data.csv', 'a') as csv_file:
                csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                ret, frame = cap.read()
            
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
            
                results = pose.process(image)
            
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                right_hip=[0,0]
                try:
                
                    landmarks = results.pose_landmarks.landmark
                    right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                    left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                    left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                    right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                    left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                    right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

                    left_leg = calculate_angle(left_hip,left_knee,left_ankle)
                    right_leg = calculate_angle(right_hip,right_knee,right_ankle)

                    cv2.putText(image,str(right_leg), 
                                        tuple(np.multiply(left_hip, [640, 480]).astype(int)), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)        


                    if left_leg > 170 or right_leg > 170:
                        position='UP'
                        msg='Correct'
                    if (left_leg < 90 or right_leg < 90) and position =='UP':
                        if left_leg or right_leg <= 80:
                             msg='ERROR'
                             position="DOWN"
                             print('right:',right_leg,'left:',left_leg)
                             print(msg)
                             info = {"counter": msg}
                             csv_writer.writerow(info)
                        elif left_leg or right_leg >= 80:
                            msg='Correct'
                            position="DOWN"
                            counter +=1
                            info = {"counter": counter}
                            csv_writer.writerow(info)
                            position=None
                            msg=None
                except:
                    pass

                cv2.rectangle(image, (0,0), (70,70), (250,228,13), -1)
                cv2.putText(image, 'REPS', (15,12), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 1, cv2.LINE_AA)        
                cv2.putText(image, str(counter), 
                            (10,60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)        
                # cv2.putText(image, str(position), 
                #             tuple(np.multiply(right_hip, [640, 480]).astype(int)), 
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)        
                cv2.rectangle(image, (550,0), (640,60), (250,228,13), -1)
                cv2.putText(image,str(msg), 
                        (550,40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2, cv2.LINE_AA)
                if msg=='Correct':
                    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                            mp_drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=2), 
                                            mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2) 
                                            )               
                elif msg=='ERROR':
                    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                            mp_drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=2), 
                                            mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2) 
                                            )               
                cv2.imshow('Live Feed', image)

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

        cap.release()
        cv2.destroyAllWindows()
def plank():
    mp_drawing=mp.solutions.drawing_utils
    mp_pose=mp.solutions.pose

    fieldnames = ["counter"]
    counter = 0     

    position = None
    msg = None

    cap = cv2.VideoCapture(0)

    def calculate_angle(a,b,c):
        a = np.array(a) 
        b = np.array(b) 
        c = np.array(c) 
        
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians*180.0/np.pi)
        
        if angle >180.0:
            angle = 360-angle
            
        return angle

    # Csv file setup
    with open('data.csv', 'w') as csv_file:
        csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        csv_writer.writeheader()

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while True:
            with open('data.csv', 'a') as csv_file:
                csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                ret, frame = cap.read()
            
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
            
                results = pose.process(image)
            
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                try:
                
                    landmarks = results.pose_landmarks.landmark
                    left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                    right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                    right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                    left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                    left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                    right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                    left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                    right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                    right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                    left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                    left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                    right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]

                    left_hand = calculate_angle(left_shoulder,left_elbow,left_wrist)
                    right_hand = calculate_angle(right_shoulder,right_elbow,right_wrist)
                    left_hip_angle = calculate_angle(left_shoulder,left_hip,left_knee)
                    right_hip_angle = calculate_angle(right_shoulder,right_hip,right_knee)

                    cv2.putText(image,str(right_hand), 
                                        tuple(np.multiply(right_shoulder, [640, 480]).astype(int)), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)        
                    cv2.putText(image,str(right_hip_angle), 
                                        tuple(np.multiply(left_hip, [640, 480]).astype(int)), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)        


                    if (left_hand > 100 or right_hand > 100):
                        msg='Started'
                        t=60
                        while t:
                            if right_hip_angle > 170 or left_hip_angle > 170:
                                min = t // 60
                                sec = t % 60
                                timer='{:02d} minute {:02d} seconds'.format(min,sec)
                                info = {"counter": timer}
                                csv_writer.writerow(info)
                                sleep(1)
                                t-=1
                            else:
                                msg='ERROR'
                                info = {"counter": timer}
                                csv_writer.writerow(info)
    
                        info = {"counter": 'Done'}
                        csv_writer.writerow(info)

                except:
                    pass

                cv2.rectangle(image, (0,0), (70,70), (250,228,13), -1)
                cv2.putText(image, 'REPS', (15,12), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 1, cv2.LINE_AA)        
                cv2.putText(image, str(counter), 
                            (10,60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)        
                # cv2.putText(image, str(position), 
                #             tuple(np.multiply(right_hip, [640, 480]).astype(int)), 
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)        
                cv2.rectangle(image, (550,0), (640,60), (250,228,13), -1)
                cv2.putText(image,str(msg), 
                        (550,40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2, cv2.LINE_AA)
                if msg=='Correct':
                    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                            mp_drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=2), 
                                            mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2) 
                                            )               
                elif msg=='ERROR':
                    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                            mp_drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=2), 
                                            mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2) 
                                            )               
                cv2.imshow('Live Feed', image)

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

        cap.release()
        cv2.destroyAllWindows()
def voicemodule():
    engine = pyttsx3.init()
    num = []
    curr = None
    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[2].id)
    while True:
        with open('data.csv', 'r') as read_obj:
            csv_reader = reader(read_obj)
            for row in csv_reader:
                if len(row) > 0:
                    if row[0] != 'counter':
                        curr = row[0]
                        if curr not in num:
                            engine.say(row[0])
                            engine.runAndWait()
                            num.append(curr)
        if engine._inLoop:
            engine.endLoop()                
# def callfun(ename,c):
#     if ename=='squat':
#         t1=Thread(target=squat(c))
#         t2=Thread(target=voicemodule)

