import argparse
from dataclasses import dataclass
from typing import List

import cv2
import imutils
import matplotlib.pyplot as plt
import numpy as np

from src.paths import INPUT_PATH, OUTPUT_PATH


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


def get_contours(first_frame: np.ndarray, current_frame: np.ndarray) -> List[int]:
    # compute the absolute difference between the current frame and
    # first frame
    frame_delta = cv2.absdiff(first_frame, current_frame)
    thresh = cv2.threshold(frame_delta, 5, 255, cv2.THRESH_BINARY)[1]
    # dilate the threshold image to fill in holes, then find contours
    # on threshold image
    thresh = cv2.dilate(thresh, None, iterations=2)
    contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return imutils.grab_contours(contours)


def draw_contours(frame: np.ndarray, contours: List[int]):
    contours = filter(lambda contour: cv2.contourArea(contour) > 500, contours)
    rectangles = map(lambda contour: cv2.boundingRect(contour), contours)
    for rectangle in rectangles:
        (x, y, w, h) = rectangle
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)


def main():
    parser = argparse.ArgumentParser(description="Enter input and output video paths")
    parser.add_argument("--input_path", default=INPUT_PATH, help="Path to input raw video.")
    parser.add_argument("--output_path", default=OUTPUT_PATH, help="Path to output video with detected motion.")
    args = parser.parse_args()

    videos_metada = get_videos_metadata(args.input_path, args.output_path)

    for i in range(videos_metada.frames_number):
        frame = videos_metada.input_video.read()
        frame = frame[1]
        gray_frame = grayscale_frame(frame)
        contours = get_contours(videos_metada.first_frame, gray_frame)
        draw_contours(frame, contours)
        cv2.imshow("motion detection", frame)
        key = cv2.waitKey(1) & 0xFF
        videos_metada.output_video.write(frame)

        # if the `q` key is pressed, break from the lop
        if key == ord("q"):
            break

    # cleanup the camera and close any open windows
    videos_metada.input_video.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
