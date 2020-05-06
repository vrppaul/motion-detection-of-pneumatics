from dataclasses import dataclass
import math
import statistics
from typing import List, Set, Tuple

import cv2
import numpy as np

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
            initial_rectangle: List[int],
            initial_mass_center: Tuple[int, ...],
            current_mass_center: Tuple[int, ...],
            movement_history: List[float],
            closest_distance: int = 20
    ):
        """
        @:param initial_mass_center: Tuple[int, ...]
        Initial mass center is taken from the first observed rectangle.
        All changes will be tracked relatively to it.
        @:param current_mass_center: Tuple[int, ...]
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
        self.most_left_points: Set[Tuple[int, ...]] = {(1000000, 1000000)}
        self.most_right_points: Set[Tuple[int, ...]] = {(-1000000, -1000000)}
        self.movement_direction_history = []
        self.current_mean_position = 0
        self.last_detected_endpoint = None
        self.amount_of_runs = 1
        self.path = []

        x, y, w, h = initial_rectangle
        self.most_left_edge = x
        self.most_right_edge = x + w
        self.most_upper_edge = y
        self.most_lower_edge = y + h

    def check_if_closest_mass_distance(self, mass_center: Tuple[int, ...]):
        """
        Evaluates whether given mass_center corresponds to this
        record by measuring current_mass_center and given mass_center euclidean distance.

        :param mass_center: List[float]
        :return: bool
        """
        return self.find_distance(self.current_mass_center, mass_center) < self.closest_distance

    @staticmethod
    def find_distance(first_mc: Tuple[int, ...], second_mc: Tuple[int, ...]) -> float:
        """
        Measures euclidean between two mass centers

        :param first_mc: List[float]
        :param second_mc: List[float]
        :return: float
        """
        return math.sqrt(
            (first_mc[0] - second_mc[0]) ** 2 + (first_mc[1] - second_mc[1]) ** 2
        )

    def update_record_data(self, rectangle: List[int]):
        """
        If given mass_center corresponds to this record, this function updates
        all parameters of this record.
        :param rectangle:
        :return:
        """

        if self.current_mass_center[0] < max(self.most_left_points, key=lambda point: point[0])[0]:
            self.most_left_points.add(self.current_mass_center)
            if len(self.most_left_points) > 5:
                self.most_left_points.remove(max(self.most_left_points, key=lambda point: point[0]))
        if self.current_mass_center[0] > min(self.most_right_points, key=lambda point: point[0])[0]:
            self.most_right_points.add(self.current_mass_center)
            if len(self.most_right_points) > 5:
                self.most_right_points.remove(min(self.most_right_points, key=lambda point: point[0]))

        x, y, w, h = rectangle
        if x < self.most_left_edge:
            self.most_left_edge = x
        if x + w > self.most_right_edge:
            self.most_right_edge = x + w
        if y < self.most_upper_edge:
            self.most_upper_edge = y
        if y + h > self.most_lower_edge:
            self.most_lower_edge = y + h

    def update_movement_history(self, mass_center: Tuple[int, ...], iteration: int):
        self.current_mass_center = mass_center
        self.movement_history[iteration - 1] = Record.find_distance(self.initial_mass_center, mass_center)


class Records:
    """
    This is a simple dataclass, which contains all records and updates them.
    """
    def __init__(self, max_amount_of_recorded_runs: int):
        self.records: List[Record] = []
        self.max_amount_of_recorded_runs = max_amount_of_recorded_runs

    def update_all(self):
        """
        This function is needed to update a record, when no movement is detected,
        but nevertheless it should be updated as 0 movement.
        :return: None
        """
        for path in self.records:
            path.movement_history.append(0)

    def update_closest_or_add_new(self, rectangles: List[List[int]], iteration: int):
        """
        Takes some mass center and iteration to either update existing record or to create new
        if no close (determined by closest_distance parameter in record class) records exist.
        :param rectangles: List[List[int]]
        Rectangles, which are detected at a given iteration.
        :param iteration: int
        Number which determines at which place record should be updated
        :return: None
        """
        rectangles_mass_centers = self._associate_mass_centers_to(rectangles)
        for rect_mass_center in rectangles_mass_centers:
            found = False
            for path in self.records:
                if path.check_if_closest_mass_distance(rect_mass_center[1]):
                    if path.amount_of_runs <= self.max_amount_of_recorded_runs:
                        path.update_record_data(rect_mass_center[0])
                    path.update_movement_history(rect_mass_center[1], iteration)
                    found = True
                    break
            if not found:
                new_record_id = len(self.records)
                self.records.append(
                    Record(
                        new_record_id,
                        rect_mass_center[0],
                        rect_mass_center[1],
                        rect_mass_center[1],
                        [0] * iteration
                    )
                )

    @staticmethod
    def _associate_mass_centers_to(rectangles: List[List[int]]) -> List[Tuple[List[int], Tuple[int, ...]]]:
        return list(map(
            lambda rectangle: (rectangle, (rectangle[0] + rectangle[2] // 2, rectangle[1] + rectangle[3] // 2)),
            rectangles
        ))

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

    def draw_rectangles(self, frame: np.ndarray):
        for path in self.records:
            cv2.rectangle(
                frame,
                (path.most_left_edge, path.most_upper_edge),
                (path.most_right_edge, path.most_lower_edge),
                (0, 255, 0),
                2
            )
            cv2.circle(
                frame,
                path.current_mass_center,
                4,
                (0, 255, 0),
                -1
            )

    def draw_trajectory(self, frame: np.ndarray):
        for path in self.records:
            if path.amount_of_runs > self.max_amount_of_recorded_runs:
                most_left_point = tuple(np.median(tuple(path.most_left_points), axis=0).astype(int))
                most_right_point = tuple(np.median(tuple(path.most_right_points), axis=0).astype(int))
                coefficients = np.polyfit(
                    (most_left_point[0], most_right_point[0]),
                    (most_left_point[1], most_right_point[1]),
                    1
                )
                left_y = int(coefficients[0] * path.most_left_edge) + int(coefficients[1])
                right_y = int(coefficients[0] * path.most_right_edge) + int(coefficients[1])
                cv2.line(
                    frame,
                    (path.most_left_edge, left_y),
                    (path.most_right_edge, right_y),
                    (0, 255, 0),
                    4
                )