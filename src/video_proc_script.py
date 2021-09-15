# import libraries
import cv2
import uuid
import argparse
import numpy as np
import onnxruntime
import statistics
from collections import deque, namedtuple, Counter
from tinydb import TinyDB
from datetime import datetime
from functools import reduce
from argparse import ArgumentParser
from pathlib import Path
# logger
from loguru import logger
logger.add("log/log.log", rotation="1 week")

#import from local libraries
from gaze_tracking import GazeTracking
from db import DBClient
from headpose.src.face_detector import FaceDetector
from headpose.src.utils import draw_axis
from utils import *


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

# constants
BLINK_THRESHOLD = args['blink_threshold']
BF_TIMESTEP = args['blink_freq_timestep']
ATHLETE = args['athlete']
WRITE_STATS = args['write_stats']
VIDEO_PATH = args['video'] 
VIDEO_NAME = VIDEO_PATH.split('/')[-1].split('.')[0]
VIDEO_FILE_NAME = VIDEO_PATH.split('/')[-1]
# MEDIA_PATH = args['media_path']


class InputVideoCapture():    
    def __init__(self, path_to_dir, video_name) -> None:
        self.file_name = video_name
        self.path_to_dir = path_to_dir
        # capture video from file
        self.cap = cv2.VideoCapture(f'{self.path_to_dir}')
        # get fps of video file
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.fourcc = cv2.VideoWriter_fourcc(*'XVID') 
        self.frame_width = int(self.cap.get(3)) 
        self.frame_height = int(self.cap.get(4)) 
        self.size = (self.frame_width, self.frame_height) 
        # Below VideoWriter object will create 
        # a frame of above defined The output
        if WRITE_STATS:
            self.result = cv2.VideoWriter(f'../media_out/{self.file_name}_out.mp4',  
                                    self.fourcc, 
                                    30, 
                                    self.size) 


class BlinkingTracker():
    def __init__(self, gaze: GazeTracking) -> None:
        self.gaze = gaze
        self.blink = False
        self.blinks_bag = deque()
        self.prev_blink = 0
        self.blink_counter = 0
        self.blink_duration = 0
        self.blink_counter_duration = 0

    def is_blinking(self):
        self.blink = self.gaze.is_blinking()

    def get_blink_freq(self, frame_counter):
        if frame_counter <= int(BF_TIMESTEP * round(video.fps,0)):
            if self.blink:
                if self.prev_blink == 0:
                    self.prev_blink = 1
                    self.blink_counter += 1
                    self.blinks_bag.appendleft(1)
                else:
                    self.blinks_bag.appendleft(0)
            else:
                self.blinks_bag.appendleft(0)
                self.prev_blink = 0
        else:
            if self.blink:
                if self.prev_blink == 0:
                    self.prev_blink = 1
                    self.blink_counter += 1
                    self.blinks_bag.appendleft(1)
                else:
                    self.blinks_bag.appendleft(0)
            else:
                self.blinks_bag.appendleft(0)
                self.prev_blink = 0
            self.blinks_bag.pop()
        
        return self.blinks_bag.count(1)

    def get_blink_duration(self):
        if self.blink:
            self.blink_duration = 0
            self.blink_counter_duration += 1
            self.blink_duration = (self.blink_counter_duration / video.fps) * 1000
        else:
            self.blink_counter_duration = 0
            self.blink_duration = 0

        return self.blink_duration


class HeadPoseAnalyzer():

    def __init__(self) -> None:
        self.yaw_acc = []
        self.pitch_acc = []
        self.roll_acc = []
        self.yaw_categ_acc = Counter()
        self.roll_categ_acc = Counter()
        self.pitch_categ_acc = Counter()
        # self.yaw_categ_acc = {'Pro': 0, 'AR': 0, 'Amateur': 0, 'Beginner': 0}
        # self.roll_categ_acc = {'Pro': 0, 'AR': 0, 'Amateur': 0, 'Beginner': 0}
        # self.pitch_categ_acc = {'Pro': 0, 'AR': 0, 'Amateur': 0, 'Beginner': 0}
        ### Initialize headpose-estimator
        self.face_d = FaceDetector()
        self.sess = onnxruntime.InferenceSession(f'headpose/pretrained/fsanet-1x1-iter-688590.onnx')
        self.sess2 = onnxruntime.InferenceSession(f'headpose/pretrained/fsanet-var-iter-688590.onnx')
        logger.info("ONNX models loaded")

    def get_axis_data(self, face_bb, frame):
        for (x1,y1,x2,y2) in face_bb:
            self.x1 = x1
            self.x2 = x2
            self.y1 = y1
            self.y2 = y2

            face_roi = frame[y1:y2+1,x1:x2+1]

            #preprocess headpose model input
            face_roi = cv2.resize(face_roi,(64,64))
            face_roi = face_roi.transpose((2,0,1))
            face_roi = np.expand_dims(face_roi,axis=0)
            face_roi = (face_roi-127.5)/128
            face_roi = face_roi.astype(np.float32)

            #get headpose
            res1 = self.sess.run(
                ["output"],
                {"input": face_roi}
                )[0]
            res2 = self.sess2.run(
                ["output"],
                {"input": face_roi}
                )[0]

            logger.info(np.mean(np.vstack((res1,res2)),axis=0))

            yaw, pitch, roll = np.mean(np.vstack((res1,res2)),axis=0)

            return yaw, pitch, roll
 
    @staticmethod
    def head_compensation_mapper(value):
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


    def get_axis_variation_acc(self):

        self.yaw_acc.append(float(self.yaw))
        self.pitch_acc.append(float(self.pitch)) 
        self.roll_acc.append(float(self.roll))

        self.yaw_mean = round(statistics.pstdev(self.yaw_acc), 3)
        self.pitch_mean = round(statistics.pstdev(self.pitch_acc), 3)
        self.roll_mean = round(statistics.pstdev(self.roll_acc), 3)

        self.yaw_categ_ = self.head_compensation_mapper(self.yaw_mean)
        self.pitch_categ_ = self.head_compensation_mapper(self.pitch_mean)
        self.roll_categ_ = self.head_compensation_mapper(self.roll_mean)


        self.yaw_categ_acc.update([self.yaw_categ_])
        self.roll_categ_acc.update([self.roll_categ_])
        self.pitch_categ_acc.update([self.pitch_categ_])

        logger.info(self.yaw_acc)
        logger.info(self.yaw_mean)
        logger.info(round(self.yaw, 2))
        logger.info(round(self.pitch, 2))
        logger.info(round(self.roll, 2))

        return self.yaw_categ_acc, self.roll_categ_acc, self.pitch_categ_acc


# initializers
#db
db = DBClient("db.json")
# some definitions
test_id = uuid.uuid4().hex
dt_string = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
# initialize gaze tracking
gaze = GazeTracking(BLINK_THRESHOLD)
video = InputVideoCapture(VIDEO_PATH, VIDEO_NAME)   
# start
frame_counter = 0
blink_tracker = BlinkingTracker(gaze)
face_analysis = HeadPoseAnalyzer()
tmp_test_list = []


while True:
    # We get a new frame from the cap
    ret, frame = video.cap.read()
    logger.info(ret)

    if ret:
        frame_counter += 1
        logger.info(f'Frame: {str(frame_counter)}')

        # We send this frame to GazeTracking to analyze it
        gaze.refresh(frame)
        frame = gaze.annotated_frame()
        
        # blinking
        blink_tracker.is_blinking()
        blink_counter = blink_tracker.blink_counter
        blink_freq = blink_tracker.get_blink_freq(frame_counter)

        # gaze direction
        if gaze.is_right():
            text = "Looking right"
        elif gaze.is_left():
            text = "Looking left"
        elif gaze.is_center():
            text = "Looking center"

        left_pupil = gaze.pupil_left_coords()
        right_pupil = gaze.pupil_right_coords()

        # head pose analysis
        face_bb = face_analysis.face_d.get(frame)
        if face_bb is not None:

            yaw_categ_acc, roll_categ_acc, pitch_categ_acc = face_analysis.get_axis_data(face_bb, frame)


        #writing on frame
        if WRITE_STATS:
            cv2.putText(frame, str(round(frame_counter / video.fps, 2)) + ' seg', (30, 800), cv2.FONT_HERSHEY_DUPLEX, 1.3, (0, 200, 0), 3)
            cv2.putText(frame, text, (90, 130), cv2.FONT_HERSHEY_DUPLEX, 1.5, (100, 50, 150), 3)
            # cv2.putText(frame, "Left pupil:  " + str(left_pupil), (90, 230), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
            # cv2.putText(frame, "Right pupil: " + str(right_pupil), (90, 265), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
            cv2.putText(frame, "Blinks " + str(blink_counter), (90, 200), cv2.FONT_HERSHEY_DUPLEX, 1.5, (100, 50, 150), 3)
            cv2.putText(frame, "BF " + str(int(blink_freq)) + ' blinks/10seg', (90, 250) , cv2.FONT_HERSHEY_DUPLEX, 1.5, (100, 50, 150), 3)
            cv2.putText(frame, "Blink duration " + str(round(blink_tracker.get_blink_duration(), 2)) + 'ms', (90, 300), cv2.FONT_HERSHEY_DUPLEX, 1.5, (100, 50, 150), 3)
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



            # cv2.putText(frame, f'YAW: {str(round(yaw, 2))}', (x1, y2+50), cv2.FONT_HERSHEY_DUPLEX, 1.5, (100, 50, 150), 3)
            # cv2.putText(frame, f'PITCH: {str(round(pitch, 2))}', (x1, y2+100), cv2.FONT_HERSHEY_DUPLEX, 1.5, (100, 50, 150), 3)
            # cv2.putText(frame, f'ROLL: {str(round(roll, 2))}', (x1, y2+150), cv2.FONT_HERSHEY_DUPLEX, 1.5, (100, 50, 150), 3)
            # logger.info(type(yaw_acc[0]))



            cv2.putText(frame, f'YAW: {round(face_bb.yaw_mean, 2)} | {face_analysis.yaw}', (face_analysis.x1-150, face_analysis.y2+50), cv2.FONT_HERSHEY_DUPLEX, 1.3, (100, 50, 150), 2) 
            cv2.putText(frame, f'PITCH: {round(face_bb.pitch_mean, 2)} | {face_analysis.pitch}', (face_analysis.x1-150, face_analysis.y2+100), cv2.FONT_HERSHEY_DUPLEX, 1.3, (100, 50, 150), 2)
            cv2.putText(frame, f'ROLL: {round(face_bb.roll_mean, 2)} | {face_analysis.roll}', (face_analysis.x1-150, face_analysis.y2+150), cv2.FONT_HERSHEY_DUPLEX, 1.3, (100, 50, 150), 2)



            # cv2.putText(frame, f'YAW: {round(yaw_mean)} | {percent_elements_in_dict(yaw_categ_acc)}', (x1-150, y2+50), cv2.FONT_HERSHEY_DUPLEX, 0.8, (100, 50, 150), 2)
            # cv2.putText(frame, f'PITCH: {round(pitch_mean)} | {percent_elements_in_dict(pitch_categ_acc)}', (x1-150, y2+100), cv2.FONT_HERSHEY_DUPLEX, 0.8, (100, 50, 150), 2)
            # cv2.putText(frame, f'ROLL: {round(roll_mean)} | {percent_elements_in_dict(roll_categ_acc)}', (x1-150, y2+150), cv2.FONT_HERSHEY_DUPLEX, 0.8, (100, 50, 150), 2)

            # yaw_categ_acc, roll_categ_acc, pitch_categ_acc = face_analysis.get_axis_data()
           
            frame = draw_axis(
                frame,
                face_analysis.yaw, 
                face_analysis.pitch, 
                face_analysis.roll,
                tdx = (face_analysis.x2 - face_analysis.x1) // 2 + face_analysis.x1,
                tdy = (face_analysis.y2 - face_analysis.y1 ) // 2 + face_analysis.y1,
                size = 50
                )

            cv2.rectangle(
                frame,
                (face_analysis.x1, face_analysis.y1),
                (face_analysis.x2, face_analysis.y2),
                (0,255,0),
                2
                )


            # place results on dicts
            event = db.test_events(
                test_id = test_id,
                frame = frame_counter,
                blink = blink_tracker.blink, 
                total_acc_blinks = blink_counter, 
                ear_left = ear_left, 
                ear_right = ear_right, 
                gaze_direction = text,
                blink_freq = blink_freq,
                blink_duration = blink_tracker.get_blink_duration(),
                left_pupil = str(left_pupil),
                right_pupil = str(right_pupil),
                yaw = str(round(face_analysis.yaw, 2)),
                yaw_categ = face_analysis.yaw_categ_,
                pitch = str(round(face_analysis.pitch, 2)),
                pitch_categ = face_analysis.pitch_categ_,
                roll = str(round(face_analysis.roll, 2)),
                roll_categ = face_analysis.roll_categ_
            )
            tmp_test_list.append(dict(event._asdict()))

            meta = db.test_meta(
                test_id=test_id,
                video_file_name=VIDEO_FILE_NAME,
                datetime=dt_string,
                fps=video.fps,
                athlete=ATHLETE,
                blink_threshold=BLINK_THRESHOLD,
                bf_timestep=BF_TIMESTEP
            )

        # write frame on otput file
        if WRITE_STATS:
            video.result.write(frame)


        if cv2.waitKey(1) == 27:
            break
    else:
        break
    
# load reuslts on db
db.persist()