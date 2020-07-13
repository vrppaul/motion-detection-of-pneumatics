from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class VideosMetadata:
    input_video: cv2.VideoCapture
    output_video: cv2.VideoWriter
    first_frame: np.ndarray
    frames_number: int


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
