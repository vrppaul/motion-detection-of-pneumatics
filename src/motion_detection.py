import argparse
import pprint

import cv2

from src.utils import MAX_AMOUNT_OF_TRAINING_RUNS, OUTPUT_PATH, DISTURBED_PATH
from src.models import MotionDetector
from src.video_utils import (
    get_videos_metadata,
    grayscale_frame,
)


def main():
    parser = argparse.ArgumentParser(description="Enter input and output video paths")
    parser.add_argument("--input_path", default=DISTURBED_PATH, help="Path to input raw video.")
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

    motion_detector = MotionDetector(
        first_frame=videos_metadata.first_frame,
        max_amount_of_recorded_runs=max_amount_of_recorded_runs
    )

    for frame_index in range(videos_metadata.frames_number):
        frame = videos_metadata.input_video.read()[1]
        gray_frame = grayscale_frame(frame)

        motion_detector.detect_motions_on_frame(gray_frame, frame_index)
        motion_detector.draw_motions_on_frame(frame)

        cv2.imshow("motion detection", frame)
        key = cv2.waitKey(1) & 0xFF
        videos_metadata.output_video.write(frame)

        # if the `q` key is pressed, break from the lop
        if key == ord("q"):
            break

    # cleanup the camera and close any open windows
    videos_metadata.input_video.release()
    cv2.destroyAllWindows()

    # movement_detector.draw_plots()
    with open("statistics.txt", "w") as statistics_file:
        statistics_file.write(pprint.pformat(motion_detector.generate_statistics()))


if __name__ == "__main__":
    main()
