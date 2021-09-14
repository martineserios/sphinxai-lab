# import extlibs
from collections import namedtuple
from tinydb import TinyDB
from datetime import datetime
import uuid

# import local libs





# some definitions
test_id = uuid.uuid4().hex
dt_string = datetime.now().strftime("%d/%m/%Y %H:%M:%S")


class DBClient():
    def __init__(self, path:str) -> None:
        # database connection
        self.db = TinyDB(path)
        self.db_tests = self.db.table('tests')
        self.db_tests_meta = self.db.table('tests_meta')
        # definition of dict to store events by frame
        self.test_events = namedtuple('event', [
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
        self.tests_meta = dict()
        self.test_meta = namedtuple('meta', [
                                                'test_id', 
                                                'video_file_name',
                                                'datetime', 
                                                'fps', 
                                                'athlete',
                                                'blink_threshold',
                                                'bf_timestep'
                                            ]
                                    )

    def persist(self):
        self.db_tests.insert_multiple(self.test_events)
        self.db_tests_meta.insert(dict(self.test_meta._asdict()))