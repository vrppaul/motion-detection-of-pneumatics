import argparse

import cv2

from src.paths import INPUT_PATH, OUTPUT_PATH
from models import Records
from video_utils import (
    get_videos_metadata,
    get_rectangles,
    grayscale_frame,
    draw_plots,
)


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
        records.update_all()
        records.update_closest_or_add_new(rectangles, i)
        records.draw_rectangles(frame)
        # records.draw_trajectory(frame)

        records.detect_and_save_direction()
        records.detect_endpoint()
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
