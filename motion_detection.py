import argparse
from dataclasses import dataclass, field
import math
import statistics
from typing import List

import cv2
import imutils
import matplotlib.pyplot as plt
import numpy as np

from src.paths import INPUT_PATH, OUTPUT_PATH

LEFT, RIGHT = 0, 1


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
            record_id: int,
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
        self.record_id = record_id
        self.initial_mass_center = initial_mass_center
        self.current_mass_center = current_mass_center
        self.movement_history = movement_history
        self.closest_distance = closest_distance
        self.most_left_point = initial_mass_center
        self.most_right_point = initial_mass_center
        self.movement_direction_history = []
        self.current_mean_position = 0
        self.last_detected_endpoint = None
        self.amount_of_runs = 1
        self.path = []

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
        if self.current_mass_center[0] < self.most_left_point[0]:
            self.most_left_point = self.current_mass_center
        elif self.current_mass_center[0] > self.most_right_point[0]:
            self.most_right_point = self.current_mass_center


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
            new_record_id = len(self.records)
            self.records.append(Record(new_record_id, mass_center, mass_center, [0] * iteration))

    def detect_and_save_direction(self):
        """
        This function detects direction by following method:
        - It takes a frame of `window_size__elements` last movements
        - It compares a median value of that frame with a median of previous window
        - If difference is bigger than `window_size__pixels` -> register a movement in a direction
        :return:
        """
        window_size__elements = 10
        window_size__pixels = 2
        for path in self.records:
            if len(path.movement_history) > window_size__elements:
                current_mean_position = statistics.median(
                    path.movement_history[-window_size__elements:]
                )
                if current_mean_position - path.current_mean_position > window_size__pixels:
                    path.movement_direction_history.append(RIGHT)
                    path.current_mean_position = current_mean_position
                elif path.current_mean_position - current_mean_position > window_size__pixels:
                    path.movement_direction_history.append(LEFT)
                    path.current_mean_position = current_mean_position

    def detect_endpoint(self):
        """
        This function detects endpoints by following method:
        - Takes a window of last 10 movement records
        - If most of records is right -> last detected endpoint is left
        - If most of records is left -> last detected endpoint is right
        - If current detected endpoint is left and last detected endpoint is right -> +1 amount of runs
        - And vice versa
        :return:
        """
        window_size = 10
        for path in self.records:
            if len(path.movement_direction_history) > window_size:
                right_directions = tuple(filter(
                    lambda direction: direction == RIGHT,
                    path.movement_direction_history[-window_size:]
                ))
                if len(right_directions) > 5:
                    if path.last_detected_endpoint == RIGHT:
                        path.amount_of_runs += 1
                    path.last_detected_endpoint = LEFT
                else:
                    if path.last_detected_endpoint == LEFT:
                        path.amount_of_runs += 1
                    path.last_detected_endpoint = RIGHT


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
        cv2.circle(frame, (x + w // 2, y + h // 2), 4, (0, 255, 0), -1)


def draw_trajectory(frame: np.ndarray, records: Records):
    for record in records.records:
        cv2.line(
            frame,
            tuple(map(lambda point: int(point), record.most_left_point)),
            tuple(map(lambda point: int(point), record.most_right_point)),
            (0, 255, 0),
            4
        )


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

    videos_metadata = get_videos_metadata(args.input_path, args.output_path)

    records = Records()

    for i in range(videos_metadata.frames_number):
        frame = videos_metadata.input_video.read()
        frame = frame[1]
        gray_frame = grayscale_frame(frame)
        rectangles = get_rectangles(videos_metadata.first_frame, gray_frame)
        draw_rectangles(frame, rectangles)
        mass_centers = list(map(
            lambda rectangle: [rectangle[0] + rectangle[2] / 2, rectangle[1] + rectangle[3] / 2],
            rectangles
        ))
        records.update_all()
        for mass_center in mass_centers:
            records.update_closest_or_add_new(mass_center, i)
        records.detect_and_save_direction()
        records.detect_endpoint()
        draw_trajectory(frame, records)
        cv2.imshow("motion detection", frame)
        key = cv2.waitKey(1) & 0xFF
        videos_metadata.output_video.write(frame)

        # if the `q` key is pressed, break from the lop
        if key == ord("q"):
            break

    # cleanup the camera and close any open windows
    videos_metadata.input_video.release()
    cv2.destroyAllWindows()

    draw_plots(records)

    print(records.records[0].amount_of_runs)
    print(records.records[0].movement_direction_history)
    print(records.records[1].amount_of_runs)
    print(records.records[1].movement_direction_history)


if __name__ == "__main__":
    main()
