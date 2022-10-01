"""
Code from https://github.com/albanie/shot-detection-benchmarks
"""

import multiprocessing
import subprocess
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm


def extract_shots_with_ffprobe(src_video, threshold=0.05):
    """
    uses ffprobe to produce a list of shot
    boundaries (in seconds)

    Args:
        src_video (string): the path to the source
            video
        threshold (float): the minimum value used
            by ffprobe to classify a shot boundary

    Returns:
        List[(float, float)]: a list of tuples of floats
        representing predicted shot boundaries (in seconds) and
        their associated scores
    """
    scene_ps = subprocess.Popen(("ffmpeg", "-i", src_video, "-vf", f"select=\'gte(scene,{threshold})\',metadata=print:",
                                 "-an", "-f", "null", "-"), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    output = str(scene_ps.stdout.read())
    boundaries = extract_boundaries_from_ffprobe_output(output)

    # Filter pairs of boundaries
    boundaries_filtered = []
    last_boundary = -1
    for boundary in boundaries:
        if boundary[1] > last_boundary + 2:  # at least 2 frames apart
            boundaries_filtered.append(boundary)
        last_boundary = boundary[1]  # we update even if not used. Probably all part of the same boundary

    if 'EKb2MMJSoeI_E_002952_003052' in src_video:
        print(boundaries, boundaries_filtered)
    return boundaries_filtered


def extract_boundaries_from_ffprobe_output(output):
    """
    extracts the shot boundaries from the string output
    producted by ffprobe

    Args:
        output (string): the full output of the ffprobe
            shot detector as a single string

    Returns:
        List[(float, int, float)]: a list of tuples of floats
        representing predicted shot boundaries (in seconds, and frame_idx) and
        their associated scores
    """
    if "Output file #0 does not contain any stream" in output:
        return []
    fps = float(output.split(' fps, ')[0].split(', ')[-1])
    boundaries = []
    metadata = output.split('\\n')[40:-3]
    lines_results = [(metadata[i], metadata[i + 1]) for i in range(0, len(metadata), 2)]
    for first_line, second_line in lines_results:
        time = float(first_line.split('pts_time:')[-1])
        frame_idx_float = time * fps
        frame_idx = int(np.round(frame_idx_float))
        # assert np.abs(frame_idx_float-frame_idx) < 0.2
        score = float(second_line.split('lavfi.scene_score=')[-1])
        boundaries.append((time, frame_idx, score))
    return boundaries


def main():
    option = 'finegym'

    dataset_name = {'finegym': 'FineGym', 'fisv': 'FisV-figure-skating', 'diving': 'Diving48'}[option]
    video_dir = {'finegym': 'event_videos', 'fisv': 'videos', 'diving': 'rgb'}[option]

    path_to_extracted_keypoints = Path('/path/to/datasets')

    path_videos = path_to_extracted_keypoints / dataset_name / video_dir
    path_save = path_to_extracted_keypoints / dataset_name / 'shot_transitions.pth'

    video_paths = list(path_videos.glob('**/*.mp4'))
    video_paths = [str(v) for v in video_paths if not v.stem.startswith('.')]

    with multiprocessing.Pool(processes=50) as pool:
        results = list(tqdm(pool.imap(extract_shots_with_ffprobe, video_paths),
                            total=len(video_paths)))
    boundaries = {video_paths[i].stem: results[i] for i in range(len(results))}

    # This commented-out code is useful to debug -- hard to do with multiprocessing
    # boundaries = {}
    # for path in tqdm(video_paths, total=len(video_paths)):
    #     boundaries_ = extract_shots_with_ffprobe(str(path))
    #     boundaries[path.stem] = boundaries_

    torch.save(boundaries, path_save)


if __name__ == '__main__':
    main()
