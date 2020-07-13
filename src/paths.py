import os

PROJECT_ROOT_PATH = os.path.abspath(os.path.join(__file__, '..', '..'))
DATA_PATH = os.path.join(PROJECT_ROOT_PATH, 'data')

INPUT_PATH = os.path.join(DATA_PATH, "render.avi")
OUTPUT_PATH = os.path.join(DATA_PATH, "motion_capture.mp4")
