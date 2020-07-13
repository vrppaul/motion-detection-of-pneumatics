import math
import statistics
from enum import Enum
from typing import Callable, List, Optional, Set, Tuple, Dict

import cv2
import imutils
import matplotlib.pyplot as plt
import numpy as np

from constants import (
    AMOUNT_OF_BORDER_POINTS,
    DIRECTION_WINDOW_SIZE__ELEMENTS,
    DIRECTION_WINDOW_SIZE__PIXELS,
    DISTANCE_WINDOW_SIZE__PIXELS,
    ENDPOINT_WINDOW_SIZE__ELEMENTS, THRESHOLD_VALUE,
)


class Direction(Enum):
    LEFT = "left"
    RIGHT = "right"
    UP = "up"
    DOWN = "down"
    LEFT_UP = "left up"
    LEFT_DOWN = "left down"
    RIGHT_UP = "right up"
    RIGHT_DOWN = "right down"


Rectangle = List[int]
Point = Tuple[int, ...]


class Motion:
    """
    This class is required to track mass centers of all pneumatics'
    detected rectangles of movement.
    """
    def __init__(
            self,
            record_id: int,
            initial_rectangle: Rectangle,
            initial_mass_center: Point,
            movement_history: List[float]
    ):
        """
        @:param initial_mass_center: Point
        Initial mass center is taken from the first observed rectangle.
        All changes will be tracked relatively to it.
        @:param current_mass_center: Point
        Current mass center is a parameter which helps to determine, whether given
        mass_center corresponds to this record by measuring euclidean distance.
        @:param movement_history: List[float]
        Tracking of all distances from initial_mass_center
        @:param closest_distance: int
        Parameter which determines how far can next mass_center be to correspond to this record
        """
        # Initializing passed arguments
        self.record_id = record_id
        self.initial_mass_center = initial_mass_center
        self.current_mass_center = initial_mass_center
        self.movement_history = movement_history

        # Initializing implicit arguments
        self.amount_of_runs: int = 1
        self.current_median_position: float = 0
        self.median_position_history: List[float] = []
        self.last_detected_endpoint: Optional[Direction] = None
        self.most_left_points: Set[Point] = {(1000000, 1000000)}
        self.most_right_points: Set[Point] = {(-1000000, -1000000)}
        self.mass_center_history: List[Point] = []
        self.movement_direction_history: List[Direction] = []
        self.trajectory_left: Optional[Point] = None
        self.trajectory_right: Optional[Point] = None

        x, y, w, h = initial_rectangle
        self.most_left_edge: int = x
        self.most_right_edge: int = x + w
        self.most_upper_edge: int = y
        self.most_lower_edge: int = y + h

        # Constants
        self.distance_window_size_pixels: int = DISTANCE_WINDOW_SIZE__PIXELS
        self.amount_of_border_points: int = AMOUNT_OF_BORDER_POINTS

    def add_zero_placeholder(self):
        self.movement_history.append(0)

    def check_if_closest_mass_distance(self, mass_center: Point):
        """
        Evaluates whether given mass_center corresponds to this
        record by measuring current_mass_center and given mass_center euclidean distance.

        :param mass_center: List[float]
        :return: bool
        """
        return self.find_distance(self.current_mass_center, mass_center) < self.distance_window_size_pixels

    def detect_and_save_direction(self, window_size_elements: int, window_size_pixels: int):
        if len(self.movement_history) > window_size_elements:
            current_median_position = statistics.median(
                self.movement_history[-window_size_elements:]  # Last elements
            )
            self.median_position_history.append(current_median_position)
            if current_median_position - self.current_median_position > window_size_pixels:
                self.movement_direction_history.append(Direction.RIGHT)
                self.current_median_position = current_median_position
            elif self.current_median_position - current_median_position > window_size_pixels:
                self.movement_direction_history.append(Direction.LEFT)
                self.current_median_position = current_median_position

    def detect_and_save_endpoint(self, window_size_elements: int):
        if len(self.movement_direction_history) > window_size_elements:
            right_directions = tuple(filter(
                lambda direction: direction == Direction.RIGHT,
                self.movement_direction_history[-window_size_elements:]
            ))
            if len(right_directions) > math.ceil(window_size_elements / 2):  # Most of records
                if self.last_detected_endpoint == Direction.RIGHT:
                    self.amount_of_runs += 1
                self.last_detected_endpoint = Direction.LEFT
            else:
                if self.last_detected_endpoint == Direction.LEFT:
                    self.amount_of_runs += 1
                self.last_detected_endpoint = Direction.RIGHT

    @staticmethod
    def find_distance(first_mc: Point, second_mc: Point) -> float:
        """
        Measures euclidean between two mass centers

        :param first_mc: List[float]
        :param second_mc: List[float]
        :return: float
        """
        return math.sqrt(
            (first_mc[0] - second_mc[0]) ** 2 + (first_mc[1] - second_mc[1]) ** 2
        )

    def update_movement_history(self, mass_center: Point, iteration: int):
        self.mass_center_history.append(mass_center)
        self.current_mass_center = mass_center
        self.movement_history[iteration - 1] = Motion.find_distance(self.initial_mass_center, mass_center)

    def update_record_data(self, rectangle: Rectangle):
        """
        If given mass_center corresponds to this record, this function updates
        all parameters of this record.
        :param rectangle:
        :return:
        """
        self._update_border_points()
        self._update_edges(rectangle)

    def calculate_and_save_trajectory(self):
        most_left_point = tuple(np.median(tuple(self.most_left_points), axis=0).astype(int))
        most_right_point = tuple(np.median(tuple(self.most_right_points), axis=0).astype(int))
        coefficients = np.polyfit(
            (most_left_point[0], most_right_point[0]),
            (most_left_point[1], most_right_point[1]),
            1
        )
        left_y = coefficients[0] * self.most_left_edge + coefficients[1]
        left_y = min(max(left_y, self.most_upper_edge), self.most_lower_edge)
        right_y = coefficients[0] * self.most_right_edge + coefficients[1]
        right_y = min(max(right_y, self.most_upper_edge), self.most_lower_edge)
        left_x = (left_y - coefficients[1]) / coefficients[0]
        right_x = (right_y - coefficients[1]) / coefficients[0]

        self.trajectory_left = [int(left_x), int(left_y)]
        self.trajectory_right = [int(right_x), int(right_y)]

    def generate_statistics(self) -> Dict:
        movement_statistics = {
            "id": self.record_id,
            "amount_of_runs": self.amount_of_runs,
            "movement_history": self._get_cleared_movement_history_for_statistics(),
            "trajectory": [self.trajectory_left, self.trajectory_right],
        }
        return movement_statistics

    def _get_cleared_movement_history_for_statistics(self):
        cleaned_movement_direction_history = [[self.movement_direction_history[0], 1]]
        for movement_direction in self.movement_direction_history[1:]:
            if movement_direction == cleaned_movement_direction_history[-1][0]:
                cleaned_movement_direction_history[-1][1] += 1
            else:
                cleaned_movement_direction_history.append([movement_direction, 1])
        return cleaned_movement_direction_history

    def _update_border_points(self):
        def _get_extreme_from_border_points(min_max_callback: Callable, points: Set[Point]) -> Point:
            return min_max_callback(points, key=lambda point: point[0])

        max_of_most_left_points = _get_extreme_from_border_points(max, self.most_left_points)
        min_of_most_right_points = _get_extreme_from_border_points(min, self.most_right_points)

        if self.current_mass_center[0] < max_of_most_left_points[0]:
            self.most_left_points.add(self.current_mass_center)
            if len(self.most_left_points) > self.amount_of_border_points:
                self.most_left_points.remove(max_of_most_left_points)
        if self.current_mass_center[0] > min_of_most_right_points[0]:
            self.most_right_points.add(self.current_mass_center)
            if len(self.most_right_points) > self.amount_of_border_points:
                self.most_right_points.remove(min_of_most_right_points)

    def _update_edges(self, rectangle: Rectangle):
        x, y, w, h = rectangle
        if x < self.most_left_edge:
            self.most_left_edge = x
        if x + w > self.most_right_edge:
            self.most_right_edge = x + w
        if y < self.most_upper_edge:
            self.most_upper_edge = y
        if y + h > self.most_lower_edge:
            self.most_lower_edge = y + h


class MotionDetector:
    """
    This class contains all movement records and updates them.
    """
    def __init__(
            self,
            first_frame: np.ndarray,
            max_amount_of_recorded_runs: int
    ):
        # Initializing passed arguments
        self.first_frame = first_frame
        self.max_amount_of_recorded_runs = max_amount_of_recorded_runs

        # Initializing implicit arguments
        self.detected_motions: List[Motion] = []

        # Constants
        self._direction_window_size_elements: int = DIRECTION_WINDOW_SIZE__ELEMENTS
        self._direction_window_size_pixels: int = DIRECTION_WINDOW_SIZE__PIXELS
        self._endpoint_window_size_elements: int = ENDPOINT_WINDOW_SIZE__ELEMENTS
        self._threshold_value: int = THRESHOLD_VALUE

    def detect_motions_on_frame(
            self,
            gray_frame: np.ndarray,
            frame_index: int
    ):
        """
        This function has the following flow:
        -
        :param gray_frame:
        :param frame_index:
        :return:
        """
        self._update_all_existing_movements_with_zero_placeholder()
        self._detect_and_update_closest_or_create_new(gray_frame, frame_index)
        self._detect_and_save_directions()
        self._detect_and_save_endpoints()
        self._calculate_and_save_trajectory()

    def add_motions_to_frame(self, frame: np.ndarray):
        self._draw_rectangles_on_frame(frame)
        self._draw_trajectory_on_frame(frame)

    def draw_plots(self):
        fig = plt.figure()
        cols = 2
        rows = len(self.detected_motions) // cols + 1
        for i, movement in enumerate(self.detected_motions):
            ax = fig.add_subplot(rows, cols, i + 1)
            ax.plot(range(len(movement.median_position_history)), movement.median_position_history)
        plt.show()

    def generate_statistics(self) -> Dict:
        all_movements_statistics = {
            "amount_of_motors": len(self.detected_motions),
            "statistics": [movement.generate_statistics() for movement in self.detected_motions]
        }
        return all_movements_statistics

    def _get_rectangles(self, current_frame: np.ndarray) -> List[List[int]]:
        # compute the absolute difference between the current frame and
        # first frame
        frame_delta = cv2.absdiff(self.first_frame, current_frame)
        thresh = cv2.threshold(frame_delta, self._threshold_value, 255, cv2.THRESH_BINARY)[1]
        # dilate the threshold image to fill in holes, then find contours
        # on threshold image
        thresh = cv2.dilate(thresh, None, iterations=2)
        contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)
        contours = filter(lambda contour: cv2.contourArea(contour) > 500, contours)
        rectangles = list(map(lambda contour: cv2.boundingRect(contour), contours))
        return rectangles

    def _update_all_existing_movements_with_zero_placeholder(self):
        """
        This function is needed to update a record, when no movement is detected,
        but nevertheless it should be updated as 0 movement.
        :return: None
        """
        for movement in self.detected_motions:
            movement.add_zero_placeholder()

    def _detect_and_update_closest_or_create_new(self, gray_frame: np.ndarray, iteration: int):
        """
        Takes some mass center and iteration to either update existing record or to create new
        if no close (determined by closest_distance parameter in record class) records exist.
        :param gray_frame: np.ndarray
        Fray frame from which rectangles will be extracted.
        :param iteration: int
        Number which determines at which place record should be updated
        :return: None
        """
        rectangles_mass_centers = self._get_rectangles_mass_centers(gray_frame)
        for rect_mass_center in rectangles_mass_centers:
            found = self._existing_movement_found_and_updated(rect_mass_center, iteration)
            if not found:
                self._create_new_movement(rect_mass_center, iteration)

    def _existing_movement_found_and_updated(
            self,
            rect_mass_center: Tuple[Rectangle, Point],
            iteration: int
    ) -> bool:
        """
        Check all existing movements. Return True if given mass center corresponds to any movement.
        Otherwise return False
        :param rect_mass_center:
        :param iteration:
        :return: bool
        """
        for movement in self.detected_motions:
            if movement.check_if_closest_mass_distance(rect_mass_center[1]):
                if movement.amount_of_runs <= self.max_amount_of_recorded_runs:
                    movement.update_record_data(rect_mass_center[0])
                movement.update_movement_history(rect_mass_center[1], iteration)
                return True
        return False

    def _create_new_movement(self, rect_mass_center: Tuple[Rectangle, Point], iteration: int):
        """
        If no existing movements were detected, create new movement
        :param rect_mass_center:
        :param iteration:
        :return:
        """
        new_record_id = len(self.detected_motions)
        self.detected_motions.append(
            Motion(
                new_record_id,
                rect_mass_center[0],
                rect_mass_center[1],
                [0] * iteration
            )
        )

    def _get_rectangles_mass_centers(self, gray_frame: np.ndarray) -> List[Tuple[Rectangle, Point]]:
        rectangles = self._get_rectangles(gray_frame)
        return list(map(
            lambda rectangle: (rectangle, (rectangle[0] + rectangle[2] // 2, rectangle[1] + rectangle[3] // 2)),
            rectangles
        ))

    def _detect_and_save_directions(self):
        """
        This function detects direction by following method:
        - It takes a frame of last `WINDOW_SIZE__ELEMENTS` movements
        - It compares a median value of that frame with a median of previous window
        - If difference is bigger than `WINDOW_SIZE__PIXELS` -> register a movement in a direction
        :return:
        """
        for movement in self.detected_motions:
            movement.detect_and_save_direction(
                self._direction_window_size_elements,
                self._direction_window_size_pixels
            )

    def _detect_and_save_endpoints(self):
        """
        This function detects endpoints by following method:
        - Takes a window of last `ENDPOINT_WINDOW_SIZE__ELEMENTS` movement records
        - If most of records is right -> last detected endpoint is left
        - If most of records is left -> last detected endpoint is right
        - If current detected endpoint is left and last detected endpoint is right -> +1 amount of runs
        - And vice versa
        :return:
        """
        for movement in self.detected_motions:
            movement.detect_and_save_endpoint(self._endpoint_window_size_elements)

    def _calculate_and_save_trajectory(self):
        for movement in self.detected_motions:
            if movement.amount_of_runs == self.max_amount_of_recorded_runs + 1:
                movement.calculate_and_save_trajectory()

    def _draw_rectangles_on_frame(self, frame: np.ndarray):
        for movement in self.detected_motions:
            cv2.rectangle(
                frame,
                (movement.most_left_edge, movement.most_upper_edge),
                (movement.most_right_edge, movement.most_lower_edge),
                (0, 255, 0),
                2
            )
            cv2.circle(
                frame,
                movement.current_mass_center,
                4,
                (0, 255, 0),
                -1
            )

    def _draw_trajectory_on_frame(self, frame: np.ndarray):
        for movement in self.detected_motions:
            if movement.amount_of_runs > self.max_amount_of_recorded_runs:
                cv2.line(
                    frame,
                    (movement.trajectory_left[0], movement.trajectory_left[1]),
                    (movement.trajectory_right[0], movement.trajectory_right[1]),
                    (0, 255, 0),
                    4
                )
