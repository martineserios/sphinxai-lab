# import libraries
from collections import deque, namedtuple, Counter
import cv2
from tinydb import TinyDB
from datetime import datetime
import uuid
from functools import reduce
import argparse

from loguru import logger
logger.add("log/log.log", rotation="1 week")

#import from local libraries
from gaze_tracking import GazeTracking

### Headpose estimator
from argparse import ArgumentParser
import numpy as np
import cv2
import onnxruntime
import sys
from pathlib import Path
import statistics
#local imports
from headpose.src.face_detector import FaceDetector
from headpose.src.utils import draw_axis

########
BLINK_THRESHOLD = 5

########


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument(
    "-bt", 
    "--blink-threshold",
    type=float,
    default=BLINK_THRESHOLD,
    help="EAR limit value to consider a blink")
ap.add_argument(
    "-bfts", 
    "--blink-freq-timestep", 
    type=int, 
    default=10,
    help="timestep to consider for freq calc")
ap.add_argument(
    "-a", 
    "--athlete", 
    type=str, 
    required=True,
    help="athlete name")
ap.add_argument(
    "-wr", 
    "--write-stats", 
    type=bool, 
    default=True,
    help="write events on the output file")
ap.add_argument(
    "-v", 
    "--video", 
    type=str, 
    required=True,
    help="video file name")
# ap.add_argument(
#     "-mp", 
#     "--media-path", 
#     type=str, 
#     default='../media/',
#     help="path to media folder")

args = vars(ap.parse_args())

# env vars
BLINK_THRESHOLD = args['blink_threshold']
BF_TIMESTEP = args['blink_freq_timestep']
ATHLETE = args['athlete']
WRITE_STATS = args['write_stats']
VIDEO_PATH = args['video'] 
VIDEO_NAME = VIDEO_PATH.split('/')[-1].split('.')[0]
VIDEO_FILE_NAME = VIDEO_PATH.split('/')[-1]
# MEDIA_PATH = args['media_path']

def HEAD_COMPENSATION_MAP(value):
    value = abs(value)
    category = ''

    if value <= abs(4):
        category = 'PRO'
    elif (value > abs(4)) & (value <= abs(8)):
        category = 'AR'
    elif (value > abs(8)) & (value <= abs(16)):
        category = 'AM'
    elif (value > abs(16)):
        category = 'BEG'
    return category

def percent_elements_in_dict(categs_dict):
    total = sum(categs_dict.values())
    percent = {key: f'{round(100 * (value/total))}%' for key, value in categs_dict.items()}
    
    return percent


def list_mean(lst):
    return reduce(lambda a, b: a + b, lst) / len(lst)

# database connection
db = TinyDB('db.json')
db_tests = db.table('tests')
db_tests_meta = db.table('tests_meta')


# some definitions
test_id = uuid.uuid4().hex
dt_string = datetime.now().strftime("%d/%m/%Y %H:%M:%S")

# definition of dict to store events by frame
test_events = namedtuple(
    'event', 
    [
        'test_id',
        'frame',
        'blink', 
        'total_acc_blinks', 
        'ear_left', 
        'ear_right', 
        'gaze_direction',
        'blink_freq',
        'blink_duration',
        'left_pupil',
        'right_pupil',
        'yaw',
        'yaw_categ',
        'pitch',
        'pitch_categ',
        'roll',
        'roll_categ'
    ]
)
tmp_test_list = []

tests_meta = {}
test_meta = namedtuple(
    'meta', 
    [
        'test_id', 
        'video_file_name',
        'datetime', 
        'fps', 
        'athlete',
        'blink_threshold',
        'bf_timestep'
    ]
)

### Initialize headpose-estimator
face_d = FaceDetector()

sess = onnxruntime.InferenceSession(f'headpose/pretrained/fsanet-1x1-iter-688590.onnx')

sess2 = onnxruntime.InferenceSession(f'headpose/pretrained/fsanet-var-iter-688590.onnx')

print("ONNX models loaded")

# initialize gaze tracking
gaze = GazeTracking(BLINK_THRESHOLD)

# capture video from file
# cap = cv2.VideoCapture(f'{MEDIA_PATH}{VIDEO_NAME}')
cap = cv2.VideoCapture(f'{VIDEO_PATH}')

# get fps of video file
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'XVID') 

# We need to set resolutions. 
# so, convert them from float to integer. 
frame_width = int(cap.get(3)) 
frame_height = int(cap.get(4)) 
size = (frame_width, frame_height) 
   
# Below VideoWriter object will create 
# a frame of above defined The output
if WRITE_STATS:
    result = cv2.VideoWriter(f'../media_out/{VIDEO_NAME}_out.mp4',  
                            fourcc, 
                            30, 
                            size) 



# definition of vars for counting events occurence
blink_counter = 0
prev_blink = 0
blink_counter_duration = 0
blink_duration = 0
frame_counter = 0

# blinks frequency queue
blinks_bag = deque()

yaw_acc = []
roll_acc = []
pitch_acc = []

yaw_categ_acc = {'Pro': 0, 'AR': 0, 'Amateur': 0, 'Beginner': 0}
roll_categ_acc = {'Pro': 0, 'AR': 0, 'Amateur': 0, 'Beginner': 0}
pitch_categ_acc = {'Pro': 0, 'AR': 0, 'Amateur': 0, 'Beginner': 0}

yaw_categ_acc = Counter()
roll_categ_acc = Counter()
pitch_categ_acc = Counter()


# start
while True:
    # We get a new frame from the cap
    ret, frame = cap.read()
    logger.info(ret)

    if ret:
        frame_counter += 1
        logger.info(f'Frame: {str(frame_counter)}')

        # We send this frame to GazeTracking to analyze it
        gaze.refresh(frame)

        frame = gaze.annotated_frame()
        text = ""

        # check if there is a blink in the frame
        blink = gaze.is_blinking()

        # blink and blink freq counter 
        if frame_counter <= int(BF_TIMESTEP * round(fps,0)):
            if blink:
                if prev_blink == 0:
                    prev_blink = 1
                    blink_counter += 1
                    blinks_bag.appendleft(1)
                else:
                    blinks_bag.appendleft(0)

            else:
                blinks_bag.appendleft(0)
                prev_blink = 0
        
        else:
            if blink:
                if prev_blink == 0:
                    prev_blink = 1
                    blink_counter += 1
                    blinks_bag.appendleft(1)
        
                else:
                    blinks_bag.appendleft(0)

            else:
                blinks_bag.appendleft(0)
                prev_blink = 0
            
            blinks_bag.pop()

        bf = blinks_bag.count(1)



        # blink duration
        if blink:
            blink_duration=0
            blink_counter_duration += 1
            blink_duration = (blink_counter_duration / fps) * 1000
        else:
            blink_counter_duration=0

        # gaze direction
        if gaze.is_right():
            text = "Looking right"
        elif gaze.is_left():
            text = "Looking left"
        elif gaze.is_center():
            text = "Looking center"

        left_pupil = gaze.pupil_left_coords()
        right_pupil = gaze.pupil_right_coords()

        #writing on frame
        if WRITE_STATS:
            cv2.putText(frame, str(round(frame_counter / fps, 2)) + ' seg', (30, 800), cv2.FONT_HERSHEY_DUPLEX, 1.3, (0, 200, 0), 3)
            cv2.putText(frame, text, (90, 130), cv2.FONT_HERSHEY_DUPLEX, 1.5, (100, 50, 150), 3)
            # cv2.putText(frame, "Left pupil:  " + str(left_pupil), (90, 230), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
            # cv2.putText(frame, "Right pupil: " + str(right_pupil), (90, 265), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
            cv2.putText(frame, "Blinks " + str(blink_counter), (90, 200), cv2.FONT_HERSHEY_DUPLEX, 1.5, (100, 50, 150), 3)
            cv2.putText(frame, "BF " + str(int(bf)) + ' blinks/10seg', (90, 250) , cv2.FONT_HERSHEY_DUPLEX, 1.5, (100, 50, 150), 3)
            cv2.putText(frame, "Blink duration " + str(round(blink_duration, 2)) + 'ms', (90, 300), cv2.FONT_HERSHEY_DUPLEX, 1.5, (100, 50, 150), 3)
            if gaze.pupils_located:
                ear_left = str(round(gaze.blinking_ratio()[0], 1))
                ear_right = str(round(gaze.blinking_ratio()[1], 1))
                cv2.putText(frame, "EAR: Left: " + str(round(gaze.blinking_ratio()[0], 1)), (90, 350), cv2.FONT_HERSHEY_DUPLEX, 1.5, (100, 50, 150), 3)
                cv2.putText(frame, "EAR: Right: " + str(round(gaze.blinking_ratio()[1], 1)), (90, 400), cv2.FONT_HERSHEY_DUPLEX, 1.5, (100, 50, 150), 3)
            else:
                ear_left = ''
                ear_right = ''
        else:
            if gaze.pupils_located:
                ear_left = str(round(gaze.blinking_ratio()[0], 1))
                ear_right = str(round(gaze.blinking_ratio()[1], 1))
            else:
                ear_left = ''
                ear_right = ''

        ### HEADPOSE ESTIMAtor
        face_bb = face_d.get(frame)
        for (x1,y1,x2,y2) in face_bb:
            face_roi = frame[y1:y2+1,x1:x2+1]

            #preprocess headpose model input
            face_roi = cv2.resize(face_roi,(64,64))
            face_roi = face_roi.transpose((2,0,1))
            face_roi = np.expand_dims(face_roi,axis=0)
            face_roi = (face_roi-127.5)/128
            face_roi = face_roi.astype(np.float32)

            #get headpose
            res1 = sess.run(["output"], {"input": face_roi})[0]
            res2 = sess2.run(["output"], {"input": face_roi})[0]

            logger.info(np.mean(np.vstack((res1,res2)),axis=0))


            yaw,pitch,roll = np.mean(np.vstack((res1,res2)),axis=0)

            yaw_acc.append(float(yaw))
            pitch_acc.append(float(yaw)) 
            roll_acc.append(float(roll))


            # cv2.putText(frame, f'YAW: {str(round(yaw, 2))}', (x1, y2+50), cv2.FONT_HERSHEY_DUPLEX, 1.5, (100, 50, 150), 3)
            # cv2.putText(frame, f'PITCH: {str(round(pitch, 2))}', (x1, y2+100), cv2.FONT_HERSHEY_DUPLEX, 1.5, (100, 50, 150), 3)
            # cv2.putText(frame, f'ROLL: {str(round(roll, 2))}', (x1, y2+150), cv2.FONT_HERSHEY_DUPLEX, 1.5, (100, 50, 150), 3)
            # logger.info(type(yaw_acc[0]))
            yaw_mean = round(statistics.pstdev(yaw_acc), 3)
            pitch_mean = round(statistics.pstdev(pitch_acc), 3)
            roll_mean = round(statistics.pstdev(roll_acc), 3)

            yaw_categ_ = HEAD_COMPENSATION_MAP(yaw_mean)
            pitch_categ_ = HEAD_COMPENSATION_MAP(pitch_mean)
            roll_categ_ = HEAD_COMPENSATION_MAP(roll_mean)

            yaw_categ_acc.update([yaw_categ_])
            roll_categ_acc.update([roll_categ_])
            pitch_categ_acc.update([pitch_categ_])

            logger.info(yaw_acc)
            logger.info(yaw_mean)


            cv2.putText(frame, f'YAW: {round(yaw_mean, 2)} | {yaw}', (x1-150, y2+50), cv2.FONT_HERSHEY_DUPLEX, 1.3, (100, 50, 150), 2) 
            cv2.putText(frame, f'PITCH: {round(pitch_mean, 2)} | {pitch}', (x1-150, y2+100), cv2.FONT_HERSHEY_DUPLEX, 1.3, (100, 50, 150), 2)
            cv2.putText(frame, f'ROLL: {round(roll_mean, 2)} | {roll}', (x1-150, y2+150), cv2.FONT_HERSHEY_DUPLEX, 1.3, (100, 50, 150), 2)



            # cv2.putText(frame, f'YAW: {round(yaw_mean)} | {percent_elements_in_dict(yaw_categ_acc)}', (x1-150, y2+50), cv2.FONT_HERSHEY_DUPLEX, 0.8, (100, 50, 150), 2)
            # cv2.putText(frame, f'PITCH: {round(pitch_mean)} | {percent_elements_in_dict(pitch_categ_acc)}', (x1-150, y2+100), cv2.FONT_HERSHEY_DUPLEX, 0.8, (100, 50, 150), 2)
            # cv2.putText(frame, f'ROLL: {round(roll_mean)} | {percent_elements_in_dict(roll_categ_acc)}', (x1-150, y2+150), cv2.FONT_HERSHEY_DUPLEX, 0.8, (100, 50, 150), 2)

            frame = draw_axis(frame,yaw,pitch,roll,tdx=(x2-x1)//2+x1,tdy=(y2-y1)//2+y1,size=50)

            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)


                
            logger.info(round(yaw, 2))
            logger.info(round(pitch, 2))
            logger.info(round(roll, 2))
            
            # place results on dicts
            event = test_events(
                test_id=test_id,
                frame=frame_counter,
                blink=blink, 
                total_acc_blinks=blink_counter, 
                ear_left=ear_left, 
                ear_right=ear_right, 
                gaze_direction=text,
                blink_freq=bf,
                blink_duration=blink_duration,
                left_pupil=str(left_pupil),
                right_pupil=str(right_pupil),
                yaw=str(round(yaw, 2)),
                yaw_categ=yaw_categ_,
                pitch=str(round(pitch, 2)),
                pitch_categ=pitch_categ_,
                roll=str(round(roll, 2)),
                roll_categ=roll_categ_
            )
            tmp_test_list.append(dict(event._asdict()))

            meta = test_meta(
                test_id=test_id,
                video_file_name=VIDEO_FILE_NAME,
                datetime=dt_string,
                fps=fps,
                athlete=ATHLETE,
                blink_threshold=BLINK_THRESHOLD,
                bf_timestep=BF_TIMESTEP
            )

        # write frame on otput file
        if WRITE_STATS:
            result.write(frame)


        if cv2.waitKey(1) == 27:
            break
    else:
        break
    
# load reuslts on db
db_tests.insert_multiple(tmp_test_list)
db_tests_meta.insert(dict(meta._asdict()))