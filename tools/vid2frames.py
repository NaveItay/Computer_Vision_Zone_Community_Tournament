import os
import argparse
import glob
import cv2
from os import path


# Parse input arges
parser = argparse.ArgumentParser()
parser.add_argument('--dir', help='Full path to vidos dir.')
parser.add_argument('--out', help='Full path to vidos dir.', default='')
args = parser.parse_args()

# Saves every save_count frame 
save_count = 5    # save img every 5 frames

# Start names from '10000.jpg'
image_name = 10000

# Save input and output directories
output_dir_path = args.out
video_dir_path = args.dir

# Run over all *.MP4 video in video_dir_path directory
# for video_file_path in glob.iglob('ChameleonVISION.MP4'):
for video_file_path in glob.iglob('DRONE-SURVEILLANCE-CONTEST-VIDEO.mp4'):

    print('Run over video -> ' + video_file_path)
    # Open Video
    vidcap = cv2.VideoCapture(video_file_path)
    # Save image
    success, image = vidcap.read()
    # Reset frame evrey new video
    frame_count = 0

    # Run over video frames
    while success:

        # Check if frame need to be saved
        if frame_count % save_count == 0:
            print("Save frame " + str(frame_count) + " as " + str(image_name) + ".jpg")
            # Save frame to image file
            cv2.imwrite(output_dir_path + str(image_name) + '.jpg', image)
            # Next image name
            image_name += 1

        # Get next frame
        success, image = vidcap.read()
        # Update frame number
        frame_count += 1
