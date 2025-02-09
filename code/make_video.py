import os
import ffmpeg
from argparse import ArgumentParser

def make_video(image_dir, output_file, fps=2):
    """Convert an image sequence to a video.

    Args:
        image_dir (str): Directory containing images.
        output_file (str): Output video file path.
        fps (int): Frames per second.
    """
    # Get sorted image list
    images = sorted(
        [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.png')],
        key=lambda x: int(x.split('_')[-1].split('.')[0])
    )
    
    if not images:
        print(f"No images found in {image_dir}")
        return

    # Create video with ffmpeg
    (
        ffmpeg
        .input(f"{image_dir}/*.png", pattern_type='glob', framerate=fps)
        .output(output_file, vcodec='libx264', pix_fmt='yuv420p')
        .run()
    )
    print(f"Video saved to {output_file}")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--image_dir", type=str, required=True, help="Image directory")
    parser.add_argument("--output", type=str, required=True, help="Output video path")
    parser.add_argument("--fps", type=int, default=2, help="Frames per second")

    args = parser.parse_args()
    
    make_video(args.image_dir, args.output, args.fps)

