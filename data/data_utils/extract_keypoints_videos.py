"""
To be executed from within the Docker container
See https://hub.docker.com/r/cwaffles/openpose
and https://github.com/CMU-Perceptual-Computing-Lab/openpose/issues/1753#issuecomment-792431838
"""

# The following is the Dockerfile to create the container.

"""Dockerfile

# https://hub.docker.com/r/cwaffles/openpose                                                                                        
FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04                                                                                      
                                                                                                                                    
#get deps                                                                                                                           
RUN apt-get update && \
DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
python3-dev python3-pip python3-setuptools git g++ wget make libprotobuf-dev protobuf-compiler libopencv-dev \
libgoogle-glog-dev libboost-all-dev libcaffe-cuda-dev libhdf5-dev libatlas-base-dev                                                 
                                                                                                                                    
#for python api                                                                                                                     
#RUN pip3 install scikit-build                                                                                                      
RUN pip3 install scikit-build numpy opencv-python==4.1.1.26 tqdm                                                                    
                                                                                                                                    
#replace cmake as old version has CUDA variable bugs                                                                                
RUN wget https://github.com/Kitware/CMake/releases/download/v3.16.0/cmake-3.16.0-Linux-x86_64.tar.gz && \
tar xzf cmake-3.16.0-Linux-x86_64.tar.gz -C /opt && \
rm cmake-3.16.0-Linux-x86_64.tar.gz                                                                                                 
ENV PATH="/opt/cmake-3.16.0-Linux-x86_64/bin:${PATH}"                                                                               
                                                                                                                                    
#get openpose                                                                                                                       
WORKDIR /openpose                                                                                                                   
RUN git clone -q --depth 1 https://github.com/CMU-Perceptual-Computing-Lab/openpose.git .                                           
                                                                                                                                    
#build it                                                                                                                           
WORKDIR /openpose/build                                                                                                             
RUN cmake -DBUILD_PYTHON=ON \
          -DDOWNLOAD_BODY_25_MODEL=ON \
          -DDOWNLOAD_BODY_MPI_MODEL=OFF \
          -DDOWNLOAD_HAND_MODEL=OFF \
          -DDOWNLOAD_FACE_MODEL=OFF \
          ..                             
                                                                                                                                    
RUN sed -ie 's/set(AMPERE "80 86")/#&/g'  ../cmake/Cuda.cmake && \
    sed -ie 's/set(AMPERE "80 86")/#&/g'  ../3rdparty/caffe/cmake/Cuda.cmake                                                        
RUN make -j`nproc`                                                                                                                  
RUN make install                                                                                                                    
                                                                                                                                    
WORKDIR /openpose                                                                                                                                                                                                                                                
                                                                                                                                    
"""

import argparse
import os
import pickle

import cv2
import numpy as np
import pyopenpose as op
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--option', type=str, default='finegym')
parser.add_argument('--split_id', type=int, default=-1)
parser.add_argument('--split_total', type=int, default=-1)
args = parser.parse_args()

options = ['skating', 'diving', 'finegym']
assert args.option in options, f'Options for --option are {options}'
# The following are paths within the Docker container
if args.option == 'skating':
    print('Skating')
    video_dir = "/home/datasets/FisV-figure-skating/videos"
    path_save_keypoints = "/home/datasets/FisV-figure-skating/keypoints"
elif args.option == 'diving':
    print('Diving')
    video_dir = "/home/datasets/Diving48/rgb"
    path_save_keypoints = "/home/datasets/Diving48/keypoints"
else:
    print('Finegym')
    video_dir = "/home/datasets/FineGym/event_videos"
    path_save_keypoints = "/home/datasets/FineGym/keypoints"

path_save = "/path/to/save/media/"
num_images_save = 5  # Save image with overlapped keypoints. Just for visualization
num_gpu = 1  # op.get_gpu_number()  # Number of GPUs.

os.makedirs(path_save_keypoints, exist_ok=True)
os.makedirs(path_save, exist_ok=True)

# Custom Params (refer to include/openpose/flags.hpp for more parameters)
params = dict(
    model_folder="/openpose/models/",
    num_gpu=int(num_gpu)
)

# Starting OpenPose
opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()

video_paths = [os.path.join(video_dir, path) for path in os.listdir(video_dir) if path.endswith('.mp4')]

if args.split_id != -1:
    assert args.split_total != -1, 'split_total cannot be -1'
    num_per_split = int(np.ceil(len(video_paths) / args.split_total))
    video_paths = [video_paths[i] for i in range(num_per_split * args.split_id, num_per_split * (args.split_id + 1))
                   if i < len(video_paths)]

for i, path_video in tqdm(enumerate(video_paths), total=len(video_paths)):
    path_save_keypoint = os.path.join(path_save_keypoints, path_video.split('/')[-1].replace('.mp4', '.pkl'))
    if os.path.isfile(path_save_keypoint):
        continue
    keypoints = []
    j = 0
    try:
        cap = cv2.VideoCapture(path_video)
        while cap.isOpened():
            # Capture frame-by-frame
            ret, frame = cap.read()
            if ret:
                j += 1
                datum = op.Datum()
                datum.cvInputData = frame
                opWrapper.emplaceAndPop(op.VectorDatum([datum]))
                keypoints_frame = datum.poseKeypoints
                keypoints.append(keypoints_frame)
                if i == 0 and j < num_images_save:
                    image = datum.cvOutputData
                    path_image = os.path.join(path_save, path_video.split('/')[-1].replace('.mp4', f'_{j}.jpg'))
                    cv2.imwrite(path_image, image)
            else:
                break
        pickle.dump(keypoints, open(path_save_keypoint, 'wb'))
        os.chmod(path_save_keypoint, 0o777)  # I am executing this as root
        # When everything done, release the video capture object
        cap.release()
        # Closes all the frames
        cv2.destroyAllWindows()
        # Process images. Single GPU because multi-GPU gave "Segmentation fault (core dumped)" error
    except Exception as e:
        print(f'Error in path {path_video}')
