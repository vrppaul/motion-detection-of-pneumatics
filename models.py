import math
import statistics
from collections import Counter
from enum import Enum
from typing import Callable, List, Optional, Set, Tuple, Dict, Any

import cv2
import imutils
import matplotlib.pyplot as plt
import numpy as np

from constants import (
    AMOUNT_OF_BORDER_POINTS,
    DIRECTION_WINDOW_SIZE__ELEMENTS,
    DIRECTION_WINDOW_SIZE__PIXELS,
    DETECT_MOTION_WINDOW_SIZE,
    ENDPOINT_WINDOW_SIZE__ELEMENTS, THRESHOLD_VALUE,
)


class Direction(Enum):
    LEFT = 2
    RIGHT = 3
    UP = 5
    DOWN = 7
    LEFT_UP = 10
    LEFT_DOWN = 14
    RIGHT_UP = 15
    RIGHT_DOWN = 21

    def is_direction(self, direction: 'Direction') -> bool:
        return self.value % direction.value == 0

    def is_left(self) -> bool:
        return self.is_direction(Direction.LEFT)

    def is_right(self) -> bool:
        return self.is_direction(Direction.RIGHT)

    def is_up(self) -> bool:
        return self.is_direction(Direction.UP)

    def is_down(self) -> bool:
        return self.is_direction(Direction.DOWN)

    def is_same(self, another: 'Direction') -> bool:
        if self.is_up() and another.is_up():
            return True
        if self.is_down() and another.is_down():
            return True
        if self.is_left() and another.is_left():
            return True
        if self.is_right() and another.is_right():
            return True

    @property
    def opposite(self) -> 'Direction':
        if self.is_left():
            return Direction.RIGHT
        elif self.is_right():
            return Direction.LEFT
        elif self.is_up():
            return Direction.DOWN
        elif self.is_down():
            return Direction.UP
        else:
            raise ValueError("Please provide the correct direction type")


class Orientation(Enum):
    VERTICAL = "vertical"
    HORIZONTAL = "horizontal"

    def is_vertical(self):
        return self is Orientation.VERTICAL

    def is_horizontal(self):
        return not self.is_vertical()


class MotionType(Enum):
    MOTOR = "motor"
    BLINKER = "blinker"


Rectangle = List[int]
Point = Tuple[int, ...]


def _find_distance(first_mc: Point, second_mc: Point) -> float:
    """
    Measures euclidean between two mass centers

    :param first_mc: List[float]
    :param second_mc: List[float]
    :return: float
    """
    return math.sqrt(
        (first_mc[0] - second_mc[0]) ** 2 + (first_mc[1] - second_mc[1]) ** 2
    )


class Motion:
    # Constants
    _amount_of_border_points: int = AMOUNT_OF_BORDER_POINTS
    _min_amount_of_rectangles_for_properties_detection: int = 10
    _detect_motion_window_size: int = DETECT_MOTION_WINDOW_SIZE
    _direction_window_size_elements: int = DIRECTION_WINDOW_SIZE__ELEMENTS
    _direction_window_size_pixels: int = DIRECTION_WINDOW_SIZE__PIXELS
    _endpoint_window_size_elements: int = ENDPOINT_WINDOW_SIZE__ELEMENTS
    max_amount_of_recorded_runs: int
    """
    This class is required to track mass centers of all pneumatics'
    detected rectangles of motion.
    """
    def __init__(
            self,
            record_id: int,
            initial_rectangle: Rectangle,
            initial_mass_center: Point,
            motion_history: List[float]
    ):
        """
        @:param initial_mass_center: Point
        Initial mass center is taken from the first observed rectangle.
        All changes will be tracked relatively to it.
        @:param current_mass_center: Point
        Current mass center is a parameter which helps to determine, whether given
        mass_center corresponds to this record by measuring euclidean distance.
        @:param motion_history: List[float]
        Tracking of all distances from initial_mass_center
        @:param closest_distance: int
        Parameter which determines how far can next mass_center be to correspond to this record
        """
        # Initializing passed arguments
        self.record_id = record_id
        self.initial_mass_center = initial_mass_center
        self.current_mass_center = initial_mass_center
        self.motion_history = motion_history

        # Initializing implicit arguments
        self.amount_of_runs: int = 1
        self.current_median_position: float = 0
        self.current_median_mass_center: Point = initial_mass_center
        self.median_position_history: List[float] = []
        self.last_detected_endpoint: Optional[Direction] = None
        self.most_left_points: Set[Point] = {(1000000, 1000000)}
        self.most_right_points: Set[Point] = {(-1000000, -1000000)}
        self.most_down_points: Set[Point] = {(-1000000, -1000000)}
        self.most_up_points: Set[Point] = {(1000000, 1000000)}
        self.mass_center_history: List[Point] = []
        self.motion_direction_history: List[Direction] = []
        self.trajectory_from: Optional[Point] = None
        self.trajectory_to: Optional[Point] = None
        self.orientation: Optional[Orientation] = None
        self.motion_type: Optional[MotionType] = None

        x, y, w, h = initial_rectangle
        self.most_left_edge: int = x
        self.most_right_edge: int = x + w
        self.most_upper_edge: int = y
        self.most_lower_edge: int = y + h

    def add_zero_placeholder(self):
        self.motion_history.append(0)

    def update_record_data(self, rectangle: Rectangle, mass_center: Point, frame_index: int):
        """
        If given mass_center corresponds to this record, this function updates
        all parameters of this record.
        :param frame_index:
        :param mass_center:
        :param rectangle:
        :return:
        """
        self._update_motion_history(mass_center, frame_index)
        self._detect_motion_type()
        if self.amount_of_runs <= Motion.max_amount_of_recorded_runs:
            self._update_edges(rectangle)
        if self.orientation is None:
            self._detect_orientation()
        else:
            if self.amount_of_runs <= Motion.max_amount_of_recorded_runs:
                self._update_border_points()
            self._detect_and_save_direction()
            self._detect_and_save_endpoint()
            if self.amount_of_runs == Motion.max_amount_of_recorded_runs:
                self._calculate_and_save_trajectory()

    def is_closest_mass_distance(self, mass_center: Point):
        """
        Evaluates whether given mass_center corresponds to this
        record by measuring current_mass_center and given mass_center euclidean distance.

        :param mass_center: List[float]
        :return: bool
        """
        return _find_distance(self.current_mass_center, mass_center) < Motion._detect_motion_window_size \
            or self._is_inside_boundaries_with_tolerance(mass_center, Motion._detect_motion_window_size)

    def _is_inside_boundaries_with_tolerance(self, mass_center: Point, tolerance: int) -> bool:
        return self.most_left_edge - tolerance < mass_center[0] < self.most_right_edge + tolerance \
            and self.most_upper_edge - tolerance < mass_center[1] < self.most_lower_edge + tolerance

    def _update_motion_history(self, mass_center: Point, frame_index: int):
        self.mass_center_history.append(mass_center)
        self.current_mass_center = mass_center
        self.motion_history[frame_index - 1] = _find_distance(self.initial_mass_center, mass_center)

    def _detect_motion_type(self):
        if len(self.mass_center_history) == Motion._min_amount_of_rectangles_for_properties_detection:
            amount_of_zero = len(list(filter(
                lambda distance: distance == 0,
                self.motion_history[-Motion._min_amount_of_rectangles_for_properties_detection:]
            )))
            if amount_of_zero > math.ceil(Motion._min_amount_of_rectangles_for_properties_detection / 2):
                self.motion_type = MotionType.BLINKER
            else:
                self.motion_type = MotionType.MOTOR

    def _detect_orientation(self):
        if self.motion_type is MotionType.BLINKER:
            return
        amount_non_null_elements = len(self.mass_center_history)
        if amount_non_null_elements == Motion._min_amount_of_rectangles_for_properties_detection:
            if self.most_right_edge - self.most_left_edge > self.most_lower_edge - self.most_upper_edge:
                self.orientation = Orientation.HORIZONTAL
            else:
                self.orientation = Orientation.VERTICAL

    def _detect_and_save_direction(self):
        if len(self.mass_center_history) > Motion._direction_window_size_elements:
            current_median_position = statistics.mean(
                self.motion_history[-Motion._direction_window_size_elements:]
            )
            current_median_x_coordinate = statistics.mean(
                [mass_center[0] for mass_center in self.mass_center_history[-Motion._direction_window_size_elements:]]
            )
            current_median_y_coordinate = statistics.mean(
                [mass_center[1] for mass_center in self.mass_center_history[-Motion._direction_window_size_elements:]]
            )
            self.median_position_history.append(current_median_position)
            if abs(current_median_position - self.current_median_position) > Motion._direction_window_size_pixels:
                diff_x = current_median_x_coordinate - self.current_median_mass_center[0]
                diff_y = current_median_y_coordinate - self.current_median_mass_center[1]

                if diff_x > 0:
                    if diff_y < 0:
                        self.motion_direction_history.append(Direction.RIGHT_UP)
                    elif diff_y == 0:
                        self.motion_direction_history.append(Direction.RIGHT)
                    else:
                        self.motion_direction_history.append(Direction.RIGHT_DOWN)
                elif diff_x == 0:
                    if diff_y < 0:
                        self.motion_direction_history.append(Direction.UP)
                    else:
                        self.motion_direction_history.append(Direction.DOWN)
                else:
                    if diff_y < 0:
                        self.motion_direction_history.append(Direction.LEFT_UP)
                    elif diff_y == 0:
                        self.motion_direction_history.append(Direction.LEFT)
                    else:
                        self.motion_direction_history.append(Direction.LEFT_DOWN)

                self.current_median_mass_center = (current_median_x_coordinate, current_median_y_coordinate)
                self.current_median_position = current_median_position

    def _detect_and_save_endpoint(self):
        """
        Imagine the following situation. Motor is going from left to right,
        thus previously detected endpoint should be LEFT (motor previously reached left side of its path).
        motion_direction_history is [RIGHT, RIGHT, RIGHT, RIGHT, RIGHT, ...]
        Then when most of elements in motion_direction_history is LEFT and last detected endpoint is LEFT as well,
        that would mean that now motor is going back and it reached the RIGHT endpoint and since it did the whole cycle
        add 1 to amount of runs
        :return:
        """
        if len(self.motion_direction_history) > Motion._endpoint_window_size_elements:
            last_directions = self.motion_direction_history[-Motion._endpoint_window_size_elements:]
            if self.orientation.is_vertical():
                transformed_directions = [
                    Direction.DOWN if direction.is_down() else Direction.UP for direction in last_directions
                ]
            else:
                transformed_directions = [
                    Direction.LEFT if direction.is_left() else Direction.RIGHT for direction in last_directions
                ]
            most_often = Counter(transformed_directions).most_common()[0][0]
            if self.last_detected_endpoint is None:
                self.last_detected_endpoint = most_often.opposite
            if self.last_detected_endpoint.is_same(most_often):
                self.amount_of_runs += 1
                self.last_detected_endpoint = most_often.opposite

    def _calculate_and_save_trajectory(self):
        if self.orientation.is_vertical():
            most_down_point, most_up_point = self._calculate_median_border_point(
                self.most_down_points, self.most_up_points
            )
            # TODO: compare to classical approach
            coefficients = np.polyfit(
                (most_down_point[0], most_up_point[0]),
                (most_down_point[1], most_up_point[1]),
                1
            )
            down_x = (self.most_lower_edge - coefficients[1]) / coefficients[0]
            down_x = min(max(down_x, self.most_left_edge), self.most_right_edge)
            up_x = (self.most_upper_edge - coefficients[1]) / coefficients[0]
            up_x = min(max(up_x, self.most_left_edge), self.most_right_edge)
            down_y = down_x * coefficients[0] + coefficients[1]
            up_y = up_x * coefficients[0] + coefficients[1]
            self.trajectory_from = (int(down_x), int(down_y))
            self.trajectory_to = (int(up_x), int(up_y))
        else:
            most_left_point, most_right_point = self._calculate_median_border_point(
                self.most_left_points, self.most_right_points
            )
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
            self.trajectory_from = (int(left_x), int(left_y))
            self.trajectory_to = (int(right_x), int(right_y))

    def _calculate_median_border_point(self, down_or_left_points: Set[Point], up_or_right_points: Set[Point]):
        from_ = tuple(np.median(tuple(down_or_left_points), axis=0).astype(int))
        to_ = tuple(np.median(tuple(up_or_right_points), axis=0).astype(int))
        return from_, to_

    def generate_statistics(self) -> Dict[str, Any]:
        motion_statistics = {
            "id": self.record_id,
            "motion_type": self.motion_type,
        }
        if self.motion_type is MotionType.MOTOR:
            specific_statistics = {
                "amount_of_runs": self.amount_of_runs,
                "motion_history": self._get_cleared_motion_history_for_statistics(),
                "trajectory": [self.trajectory_from, self.trajectory_to],
                "orientation": self.orientation,
            }
        else:
            specific_statistics = {
                "location": self.initial_mass_center
            }
        motion_statistics.update(specific_statistics)
        return motion_statistics

    def _get_cleared_motion_history_for_statistics(self):
        cleaned_motion_direction_history = [[self.motion_direction_history[0], 1]]
        for motion_direction in self.motion_direction_history[1:]:
            if motion_direction == cleaned_motion_direction_history[-1][0]:
                cleaned_motion_direction_history[-1][1] += 1
            else:
                cleaned_motion_direction_history.append([motion_direction, 1])
        return cleaned_motion_direction_history

    def _update_border_points(self):
        if self.orientation.is_vertical():
            self._update_vertical_border_points()
        else:
            self._update_horizontal_border_points()

    # FIXME: maybe better name would be left_or_up and right_or_down
    def _update_any_orientation_points(self, points_of_max: Set[Point], points_of_min: Set[Point], axis: int):
        def _get_extreme_from_border_points(min_max_callback: Callable, points: Set[Point]) -> Point:
            return min_max_callback(points, key=lambda point: point[axis])

        max_of_points = _get_extreme_from_border_points(max, points_of_max)
        min_of_points = _get_extreme_from_border_points(min, points_of_min)

        if self.current_mass_center[axis] < max_of_points[axis]:
            points_of_max.add(self.current_mass_center)
            if len(points_of_max) > Motion._amount_of_border_points:
                points_of_max.remove(max_of_points)
        if self.current_mass_center[axis] > min_of_points[axis]:
            points_of_min.add(self.current_mass_center)
            if len(points_of_min) > Motion._amount_of_border_points:
                points_of_min.remove(min_of_points)

    def _update_horizontal_border_points(self):
        self._update_any_orientation_points(self.most_left_points, self.most_right_points, 0)

    def _update_vertical_border_points(self):
        self._update_any_orientation_points(self.most_up_points, self.most_down_points, 1)

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
    _threshold_value: int = THRESHOLD_VALUE
    """
    This class contains all motion records and updates them.
    """
    def __init__(
            self,
            first_frame: np.ndarray,
            max_amount_of_recorded_runs: int
    ):
        # Initializing passed arguments
        self.first_frame = first_frame

        # Initialize the motion max amount of runs, which would be similar for all motions
        Motion.max_amount_of_recorded_runs = max_amount_of_recorded_runs

        # Initializing implicit arguments
        self.detected_motions: List[Motion] = []

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
        self._update_all_existing_motions_with_zero_placeholder()
        self._detect_and_update_existing_motions_or_create_new(gray_frame, frame_index)

    def draw_motions_on_frame(self, frame: np.ndarray):
        self._draw_rectangles_on_frame(frame)
        self._draw_trajectory_on_frame(frame)

    def draw_plots(self):
        fig = plt.figure()
        cols = 2
        rows = len(self.detected_motions) // cols + 1
        for i, motion in enumerate(self.detected_motions):
            ax = fig.add_subplot(rows, cols, i + 1)
            ax.plot(range(len(motion.median_position_history)), motion.median_position_history)
        plt.show()

    def generate_statistics(self) -> Dict:
        all_motions_statistics = {
            "amount_of_motors": len(self.detected_motions),
            "statistics": [motion.generate_statistics() for motion in self.detected_motions]
        }
        return all_motions_statistics

    def _get_rectangles(self, current_frame: np.ndarray) -> List[List[int]]:
        # compute the absolute difference between the current frame and
        # first frame
        frame_delta = cv2.absdiff(self.first_frame, current_frame)
        thresh = cv2.threshold(frame_delta, MotionDetector._threshold_value, 255, cv2.THRESH_BINARY)[1]
        # dilate the threshold image to fill in holes, then find contours
        # on threshold image
        thresh = cv2.dilate(thresh, None, iterations=2)
        contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)
        contours = filter(lambda contour: cv2.contourArea(contour) > 20, contours)
        rectangles = list(map(lambda contour: cv2.boundingRect(contour), contours))
        return rectangles

    def _update_all_existing_motions_with_zero_placeholder(self):
        """
        This function is needed to update a record, when no motion is detected,
        but nevertheless it should be updated as 0 motion.
        :return: None
        """
        for motion in self.detected_motions:
            motion.add_zero_placeholder()

    def _detect_and_update_existing_motions_or_create_new(
            self,
            gray_frame: np.ndarray,
            frame_index: int
    ):
        """
        Takes some mass center and frame_index to either update existing record or to create new
        if no close (determined by closest_distance parameter in record class) records exist.
        :param gray_frame: np.ndarray
        Fray frame from which rectangles will be extracted.
        :param frame_index: int
        Number which determines at which place record should be updated
        :return: List[Motion]
        """
        rectangles_mass_centers = self._get_rectangles_mass_centers(gray_frame)
        for rectangle, mass_center in rectangles_mass_centers:
            self._detect_and_update_single_existing_motion_or_create_new(rectangle, mass_center, frame_index)

    def _detect_and_update_single_existing_motion_or_create_new(
            self,
            rectangle: Rectangle,
            mass_center: Point,
            frame_index: int
    ):
        """
        Check all existing motions. If no existing motion was detected, create a new motion
        :param mass_center:
        :return: bool
        """
        for motion in self.detected_motions:
            if motion.is_closest_mass_distance(mass_center):
                motion.update_record_data(rectangle, mass_center, frame_index)
                return
        self._create_new_motion(rectangle, mass_center, frame_index)

    def _create_new_motion(self, rectangle: Rectangle, mass_center: Point, frame_index: int):
        """
        If no existing motions were detected, create new motion
        :param rect_mass_center:
        :param frame_index:
        :return:
        """
        new_record_id = len(self.detected_motions) + 1
        self.detected_motions.append(
            Motion(
                record_id=new_record_id,
                initial_rectangle=rectangle,
                initial_mass_center=mass_center,
                motion_history=[0]*frame_index
            )
        )

    def _get_rectangles_mass_centers(self, gray_frame: np.ndarray) -> List[Tuple[Rectangle, Point]]:
        rectangles = self._get_rectangles(gray_frame)
        return list(map(
            lambda rectangle: (rectangle, (rectangle[0] + rectangle[2] // 2, rectangle[1] + rectangle[3] // 2)),
            rectangles
        ))

    def _draw_rectangles_on_frame(self, frame: np.ndarray):
        for motion in self.detected_motions:
            cv2.rectangle(
                frame,
                (motion.most_left_edge, motion.most_upper_edge),
                (motion.most_right_edge, motion.most_lower_edge),
                (0, 255, 0),
                2
            )
            cv2.circle(
                frame,
                motion.current_mass_center,
                4,
                (0, 255, 0),
                -1
            )
            cv2.putText(
                img=frame,
                text=f"{motion.record_id}: {motion.motion_type.value if motion.motion_type else ''}",
                org=(motion.most_left_edge, motion.most_upper_edge - 10),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.3,
                color=(0, 255, 0),
                thickness=1,
                lineType=cv2.LINE_AA
            )

    def _draw_trajectory_on_frame(self, frame: np.ndarray):
        for motion in self.detected_motions:
            if motion.amount_of_runs > Motion.max_amount_of_recorded_runs:
                cv2.line(
                    frame,
                    motion.trajectory_from,
                    motion.trajectory_to,
                    (0, 255, 0),
                    4
                )
