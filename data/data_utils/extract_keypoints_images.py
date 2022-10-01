"""
To be executed from within the Docker container
See https://hub.docker.com/r/cwaffles/openpose
and https://github.com/CMU-Perceptual-Computing-Lab/openpose/issues/1753#issuecomment-792431838
"""

import os
import pickle

import cv2
import pyopenpose as op
from tqdm import tqdm

# Process a directory of images. Read all standard formats (jpg, png, bmp, etc.).

dataset_dir = "/home/datasets/EverybodyDanceNow/"
subdirectories = ['subject1', 'subject2', 'subject3', 'subject4', 'subject5', 'subject6']
path_save = "/path/to/save/media/"  # This is a path within the Docker container
num_images_save = 5  # Save image with overlapped keypoints. Just for visualization
num_gpu = 1  # op.get_gpu_number()  # Number of GPUs.

os.makedirs(path_save, exist_ok=True)

params = dict(
    model_folder="/openpose/models/",
    num_gpu=int(num_gpu)
)

# Starting OpenPose
opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()

for i, subdirectory in tqdm(enumerate(subdirectories), total=len(subdirectories)):
    for split in ['train', 'val']:
        path_save_keypoint = os.path.join(dataset_dir, subdirectory, split, 'keypoints.pkl')
        if os.path.isfile(path_save_keypoint):
            continue
        path_images = os.path.join(dataset_dir, subdirectory, split, 'test_img' if split == 'val' else 'train_img')
        list_images = os.listdir(path_images)
        keypoints = {}
        for j, im_path in tqdm(enumerate(list_images), desc=f'{subdirectory}, {split}', total=len(list_images)):
            im = cv2.imread(os.path.join(path_images, im_path))
            datum = op.Datum()
            datum.cvInputData = im
            opWrapper.poseKeypoints(op.VectorDatum([datum]))
            keypoints_frame = datum.poseKeypoints
            keypoints[im_path] = keypoints_frame
            if i == 0 and j < num_images_save:
                image = datum.cvOutputData
                path_image = os.path.join(path_save, im_path)
                cv2.imwrite(path_image, image)

        pickle.dump(keypoints, open(path_save_keypoint, 'wb'))
        os.chmod(path_save_keypoint, 0o777)  # I am executing this as root
