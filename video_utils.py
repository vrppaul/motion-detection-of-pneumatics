from typing import List

import cv2
import imutils
import matplotlib.pyplot as plt
import numpy as np

from models import VideosMetadata, Records


def grayscale_frame(frame: np.ndarray) -> np.ndarray:
    # convert frame to grayscale, and blur it
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(gray, (31, 31), 0)


def get_videos_metadata(input_path: str, output_path: str) -> VideosMetadata:
    input_video = cv2.VideoCapture(input_path)
    first_frame = input_video.read()[1]
    height, width = first_frame.shape[:2]
    first_frame = grayscale_frame(first_frame)
    fps = input_video.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_video = cv2.VideoWriter(output_path, fourcc, fps, (width, height), True)

    # -1 since we've already parsed first frame
    frames_number = int(input_video.get(cv2.CAP_PROP_FRAME_COUNT)) - 1

    return VideosMetadata(input_video, output_video, first_frame, frames_number)


def get_rectangles(first_frame: np.ndarray, current_frame: np.ndarray) -> List[List[int]]:
    # compute the absolute difference between the current frame and
    # first frame
    frame_delta = cv2.absdiff(first_frame, current_frame)
    thresh = cv2.threshold(frame_delta, 5, 255, cv2.THRESH_BINARY)[1]
    # dilate the threshold image to fill in holes, then find contours
    # on threshold image
    thresh = cv2.dilate(thresh, None, iterations=2)
    contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    contours = filter(lambda contour: cv2.contourArea(contour) > 500, contours)
    rectangles = list(map(lambda contour: cv2.boundingRect(contour), contours))
    return rectangles


def draw_plots(records: Records):
    fig = plt.figure()
    cols = 2
    rows = len(records.records) // cols + 1
    for i, path in enumerate(records.records):
        ax = fig.add_subplot(rows, cols, i + 1)
        ax.plot(range(len(path.movement_history)), path.movement_history)
    plt.show()


