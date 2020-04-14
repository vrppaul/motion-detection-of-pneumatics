"""
python disturber.py <input_file> <output_file>

python disturber.py <input_file>
- the default output is "out.avi"

"""
import argparse
import random

import cv2
import numpy as np

from src.paths import INPUT_PATH, OUTPUT_PATH


def disturb_image(frame):
    out = frame.copy()

    # shake effect
    # if random.choice([True, False]):
    #     if random.choice([True, False]):
    #         out[1:, :] = out[:-1, :]
    #     else:
    #         out[:-1, :] = out[1:, :]
    #
    # if random.choice([True, False]):
    #     if random.choice([True, False]):
    #         out[:, 1:] = out[:, :-1]
    #     else:
    #         out[:, :-1] = out[:, 1:]

    # add noise
    noise_points = np.random.randint(0, 100, out.shape, dtype=out.dtype)
    out = cv2.add(out, noise_points)

    # add line noise
    # noise_lines = np.random.choice(a=[False, True], size=out.shape[1], p=[0.99, 0.01])
    # out[:, noise_lines] = 255

    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enter input and output video paths")
    parser.add_argument("--input_path", default=INPUT_PATH, help="Path to input raw video.")
    parser.add_argument("--output_path", default=OUTPUT_PATH, help="Path to output video with detected motion.")
    args = parser.parse_args()

    # Create a VideoCapture object
    cap = cv2.VideoCapture(args.input_path)

    fps = cap.get(cv2.CAP_PROP_FPS)

    # Check if camera opened successfully
    if not cap.isOpened():
        print("Unable to read camera feed")

    # Default resolutions of the frame are obtained.The default resolutions are system dependent.
    # We convert the resolutions from float to integer.
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    # Define the codec and create VideoWriter object.The output is stored in 'out.avi' file.
    out = cv2.VideoWriter(args.output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    while True:
        ret, frame = cap.read()

        if ret:

            # Write the frame into the file 'output.avi'

            frame_processed = disturb_image(frame)

            out.write(frame_processed)

            # Display the resulting frame
            cv2.imshow('frame', frame_processed)

            # Press Q on keyboard to stop recording
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Break the loop
        else:
            break

    # When everything done, release the video capture and video write objects
    cap.release()
    out.release()

    # Closes all the frames
    cv2.destroyAllWindows()
