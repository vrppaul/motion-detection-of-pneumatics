import argparse
from pprint import pprint

import cv2

from constants import MAX_AMOUNT_OF_TRAINING_RUNS
from src.paths import INPUT_PATH, OUTPUT_PATH
from models import Movements
from video_utils import (
    get_videos_metadata,
    grayscale_frame,
)


def main():
    parser = argparse.ArgumentParser(description="Enter input and output video paths")
    parser.add_argument("--input_path", default=INPUT_PATH, help="Path to input raw video.")
    parser.add_argument("--output_path", default=OUTPUT_PATH, help="Path to output video with detected motion.")
    parser.add_argument(
        "--recorded_runs",
        default=MAX_AMOUNT_OF_TRAINING_RUNS,
        help="This number represents the maximum amount of runs,"
             "which are taken to train the model to detect the movement"
    )
    args = parser.parse_args()

    videos_metadata = get_videos_metadata(args.input_path, args.output_path)
    max_amount_of_recorded_runs = args.recorded_runs

    movements = Movements(
        first_frame=videos_metadata.first_frame,
        max_amount_of_recorded_runs=max_amount_of_recorded_runs
    )

    for frame_index in range(videos_metadata.frames_number):
        frame = videos_metadata.input_video.read()[1]
        gray_frame = grayscale_frame(frame)

        movements.detect_movement_on_frame(gray_frame, frame_index)
        movements.add_movements_to_frame(frame)

        cv2.imshow("motion detection", frame)
        key = cv2.waitKey(1) & 0xFF
        videos_metadata.output_video.write(frame)

        # if the `q` key is pressed, break from the lop
        if key == ord("q"):
            break

    # cleanup the camera and close any open windows
    videos_metadata.input_video.release()
    cv2.destroyAllWindows()

    movements.draw_plots()
    pprint(movements.generate_statistics())


if __name__ == "__main__":
    main()
