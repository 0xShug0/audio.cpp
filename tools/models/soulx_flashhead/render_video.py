#!/usr/bin/env python3
"""Encode audio.cpp's RGB24 video artifacts as an MP4."""
import argparse
import json
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True, help='CLI artifact directory')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--audio', type=Path)
    args = parser.parse_args()
    files = sorted(args.input.glob('soulx_flashhead_video_*.json'),
                   key=lambda path: int(path.stem.rsplit('_', 1)[1]))
    if not files:
        raise ValueError('No FlashHead video artifacts found')
    first = json.loads(files[0].read_text())['meta']
    args.output.parent.mkdir(parents=True, exist_ok=True)
    command = ['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pixel_format', 'rgb24',
               '-video_size', first['width'] + 'x' + first['height'], '-framerate', first['fps'], '-i', 'pipe:0']
    if args.audio:
        command += ['-i', str(args.audio), '-c:a', 'aac', '-shortest']
    command += ['-c:v', 'libx264', '-crf', '18', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(args.output)]
    with subprocess.Popen(command, stdin=subprocess.PIPE) as process:
        try:
            for file in files:
                artifact = json.loads(file.read_text())
                meta = artifact['meta']
                if any(meta[k] != first[k] for k in ('width', 'height', 'fps', 'format')) or meta['format'] != 'rgb24':
                    raise ValueError('Video artifact formats differ')
                payload = bytes.fromhex(artifact['payload_hex'])
                if len(payload) != int(meta['width']) * int(meta['height']) * int(meta['frames']) * 3:
                    raise ValueError('Video artifact byte count does not match its dimensions')
                process.stdin.write(payload)
        finally:
            process.stdin.close()
        if process.wait():
            raise RuntimeError('FFmpeg video encoding failed')


if __name__ == '__main__':
    main()
