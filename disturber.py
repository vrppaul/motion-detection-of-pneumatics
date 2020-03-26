"""
python disturber.py <input_file> <output_file>

python disturber.py <input_file>
- the default output is "out.avi"

"""
import sys
import random

import cv2
import numpy as np

def disturb_image(frame):
    out = frame.copy()

    # shake effect
    if random.choice([True, False]):
        if random.choice([True, False]):
            out[1:,:] = out[:-1,:]
        else:
            out[:-1,:] = out[1:,:]

    if random.choice([True, False]):
        if random.choice([True, False]):
            out[:,1:] = out[:,:-1]
        else:
            out[:,:-1] = out[:,1:]

    # add noise
    noise_points = np.random.randint(0, 100, out.shape, dtype=out.dtype)
    out = cv2.add(out, noise_points)

    # add line noise
    noise_lines = np.random.choice(a=[False, True], size=out.shape[1], p=[0.99, 0.01])
    out[:,noise_lines] = 255

    return out

if __name__ == "__main__":
    INPUT_PATH = sys.argv[1]
    try:
        OUTPUT_PATH = sys.argv[2]
    except:
        OUTPUT_PATH = "out.avi"


    # Create a VideoCapture object
    cap = cv2.VideoCapture(INPUT_PATH)

    fps = cap.get(cv2.CAP_PROP_FPS)


    # Check if camera opened successfully
    if (cap.isOpened() == False):
        print("Unable to read camera feed")

    # Default resolutions of the frame are obtained.The default resolutions are system dependent.
    # We convert the resolutions from float to integer.
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    # Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
    out = cv2.VideoWriter(OUTPUT_PATH, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, (frame_width, frame_height))

    while (True):
        ret, frame = cap.read()

        if ret == True:

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



