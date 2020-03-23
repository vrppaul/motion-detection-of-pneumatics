import argparse
from dataclasses import dataclass, field
import math
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


class Record:
    """
    This class is required to track mass centers of all pneumatics'
    detected rectangles of movement.
    """
    def __init__(
            self,
            initial_mass_center: List[float],
            current_mass_center: List[float],
            movement_history: List[float],
            closest_distance: int = 20
    ):
        """
        @:param initial_mass_center: List[float]
        Initial mass center is taken from the first observed rectangle.
        All changes will be tracked relatively to it.
        @:param current_mass_center: List[float]
        Current mass center is a parameter which helps to determine, whether given
        mass_center corresponds to this record by measuring euclidean distance.
        @:param movement_history: List[float]
        Tracking of all distances from initial_mass_center
        @:param closest_distance: int
        Parameter which determines how far can next mass_center be to correspond to this record
        """
        self.initial_mass_center = initial_mass_center
        self.current_mass_center = current_mass_center
        self.movement_history = movement_history
        self.closest_distance = closest_distance

    def check_if_closest_mass_distance(self, mass_center: List[float]):
        """
        Evaluates whether given mass_center corresponds to this
        record by measuring current_mass_center and given mass_center euclidean distance.

        :param mass_center: List[float]
        :return: bool
        """
        return self.find_distance(self.current_mass_center, mass_center) < self.closest_distance

    @staticmethod
    def find_distance(first_mc: List[float], second_mc: List[float]) -> float:
        """
        Measures euclidean between two mass centers

        :param first_mc: List[float]
        :param second_mc: List[float]
        :return: float
        """
        return math.sqrt(
            (first_mc[0] - second_mc[0]) ** 2 + (first_mc[1] - second_mc[1]) ** 2
        )

    def update_record_data(self, mass_center: List[float], iteration: int):
        """
        If given mass_center corresponds to this record, this function updates
        all parameters of this record.
        :param mass_center:
        :param iteration:
        :return:
        """
        self.movement_history[iteration - 1] = Record.find_distance(self.initial_mass_center, mass_center)
        self.current_mass_center = mass_center


@dataclass
class Records:
    """
    This is a simple dataclass, which contains all records and updates them.
    """
    records: List[Record] = field(default_factory=list)

    def update_all(self):
        """
        This function is needed to update a record, when no movement is detected,
        but nevertheless it should be updated as 0 movement.
        :return: None
        """
        for path in self.records:
            path.movement_history.append(0)

    def update_closest_or_add_new(self, mass_center: List[float], iteration: int):
        """
        Takes some mass center and iteration to either update existing record or to create new
        if no close (determined by closest_distance parameter in record class) records exist.
        :param mass_center: List[float]
        Some given mass center to add or update an existing record.
        :param iteration: int
        Number which determines at which place record should be updated
        :return: None
        """
        found = False
        for path in self.records:
            if path.check_if_closest_mass_distance(mass_center):
                path.update_record_data(mass_center, iteration)
                found = True
                break
        if not found:
            self.records.append(Record(mass_center, mass_center, [0] * iteration))


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


def get_rectangles(first_frame: np.ndarray, current_frame: np.ndarray) -> List[int]:
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


def draw_rectangles(frame: np.ndarray, rectangles: List[int]):
    for rectangle in rectangles:
        (x, y, w, h) = rectangle
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)


def draw_plots(records: Records):
    fig = plt.figure()
    cols = 2
    rows = len(records.records) // cols + 1
    for i, path in enumerate(records.records):
        ax = fig.add_subplot(rows, cols, i + 1)
        ax.plot(range(len(path.movement_history)), path.movement_history)
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Enter input and output video paths")
    parser.add_argument("--input_path", default=INPUT_PATH, help="Path to input raw video.")
    parser.add_argument("--output_path", default=OUTPUT_PATH, help="Path to output video with detected motion.")
    args = parser.parse_args()

    videos_metada = get_videos_metadata(args.input_path, args.output_path)

    records = Records()

    for i in range(videos_metada.frames_number):
        frame = videos_metada.input_video.read()
        frame = frame[1]
        gray_frame = grayscale_frame(frame)
        rectangles = get_rectangles(videos_metada.first_frame, gray_frame)
        draw_rectangles(frame, rectangles)
        mass_centers = list(map(
            lambda rectangle: [rectangle[0] + rectangle[2] / 2, rectangle[1] + rectangle[3] / 2],
            rectangles
        ))
        records.update_all()
        for mass_center in mass_centers:
            records.update_closest_or_add_new(mass_center, i)
        cv2.imshow("motion detection", frame)
        key = cv2.waitKey(1) & 0xFF
        videos_metada.output_video.write(frame)

        # if the `q` key is pressed, break from the lop
        if key == ord("q"):
            break

    # cleanup the camera and close any open windows
    videos_metada.input_video.release()
    cv2.destroyAllWindows()

    draw_plots(records)


if __name__ == "__main__":
    main()
