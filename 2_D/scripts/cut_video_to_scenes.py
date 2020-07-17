from glob import glob
from typing import cast, List, Tuple
import argparse
import os

from moviepy.editor import VideoFileClip
import tqdm


def cut_videos_to_scenes(args) -> None:
    paths = get_paths(args.input_videos_dir, args.input_scenes_dir)
    for video_path, scene_tags_path in tqdm.tqdm(paths):
        scene_tags = load_scene_tags(scene_tags_path)
        cut_to_scenes(video_path, args.output_dir, scene_tags)


def get_paths(videos_dir: str, scenes_dir: str) -> List[Tuple[str, str]]:
    videos_paths = sorted(glob(os.path.join(os.getcwd(), videos_dir, '*.*')))
    scenes_paths = sorted(glob(os.path.join(os.getcwd(), scenes_dir, '*.*')))
    if not are_paths_valid(videos_paths, scenes_paths):
        raise ValueError('Video files\' names or number do not match those of scene tags files.')
    return list(zip(videos_paths, scenes_paths))


def are_paths_valid(videos_paths: List[str], scenes_paths: List[str]) -> bool:
    if len(videos_paths) != len(scenes_paths):
        return False
    for video_path, scene_path in zip(videos_paths, scenes_paths):
        video_filename = os.path.splitext(os.path.split(video_path)[1])[0]
        scene_filename = os.path.splitext(os.path.split(scene_path)[1])[0]
        if video_filename != scene_filename:
            return False
    return True


def load_scene_tags(scene_tags_path: str) -> List[Tuple[int, int]]:
    with open(scene_tags_path) as fp:
        lines = fp.readlines()
    scene_tags = [tuple([int(tag) for tag in line.split(' ')]) for line in lines]
    return cast(List[Tuple[int, int]], scene_tags)


def cut_to_scenes(video_path: str, output_dir: str, scene_tags: List[Tuple[int, int]]) -> None:
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    video = VideoFileClip(video_path, verbose=False)
    for idx, (start, end) in enumerate(scene_tags):
        scene = video.subclip(start / video.fps, end / video.fps)
        video_filename = os.path.splitext(os.path.split(video_path)[1])[0]
        scene_filename = f'{video_filename}__scene_{idx}.mp4'
        scene.write_videofile(os.path.join(output_dir, scene_filename), verbose=False, logger=None)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--input-videos-dir', '-ivd', type=str, required=True, help='Directory with input videos'
    )
    parser.add_argument(
        '--input-scenes-dir', '-isd', type=str, required=True,
        help='Directory with input text files with pairs of indices of first and last frame of each scene'
    )
    parser.add_argument(
        '--output-dir', '-od', type=str, required=True, help='Directory where to save cut scenes'
    )

    args = parser.parse_args()
    cut_videos_to_scenes(args)
