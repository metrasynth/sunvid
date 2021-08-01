import ctypes
from pathlib import Path

import click
import moviepy.audio.AudioClip
import moviepy.editor
import numpy as np
import pkg_resources
from sunvox.api import INIT_FLAG, Slot, audio_callback, deinit, get_ticks, init
from tqdm import tqdm


FREQ = 48000
CHANNELS = 2
DATA_TYPE = np.float32
CDATA_TYPE = ctypes.POINTER(ctypes.c_float)

SCOPE_DATA_TYPE = np.int16
SCOPE_CDATA_TYPE = ctypes.POINTER(ctypes.c_int16)

MAX_MINUTES = 8.0
MAX_FRAMES = int(FREQ * 60 * MAX_MINUTES)

OUTFILE_TEMPLATE = "{project_path.stem}-{width}x{height}-{fps}fps.mp4"


@click.group("sunvid")
def main():
    pass


@main.command("version")
def version():
    info = pkg_resources.get_distribution("sunvid")
    print(info)


@main.command("render")
@click.argument("project-path", type=Path)
@click.option("--output-path-template", type=str, default=OUTFILE_TEMPLATE)
@click.option("--fps", type=int, default=15)
@click.option("--width", type=int, default=320)
@click.option("--height", type=int, default=180)
@click.option("--font", type=str, default="SunDogMedium")
@click.option("--audio-bitrate", type=int, default=96)
@click.option("--video-bitrate", type=int, default=32)
@click.option("--audio-codec", type=str, default="aac")
@click.option("--video-codec", type=str, default="libx264")
@click.option("--overwrite", type=bool, is_flag=True, default=False)
def render(
    project_path: Path,
    output_path_template: str,
    fps: int,
    width: int,
    height: int,
    font: str,
    audio_bitrate: int,
    video_bitrate: int,
    audio_codec: str,
    video_codec: str,
    overwrite: bool,
):
    project_path = project_path.absolute()
    if not project_path.exists():
        raise FileNotFoundError(f"{project_path} not found")
    output_path = project_path.parent / output_path_template.format(**locals())
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"{output_path} already exists")
    if FREQ / fps != (audio_frames_per_video_frame := int(FREQ / fps)):
        raise ValueError(f"{FREQ} not evenly divisible by {fps}")
    init(
        None,
        FREQ,
        2,
        INIT_FLAG.AUDIO_FLOAT32
        | INIT_FLAG.ONE_THREAD
        | INIT_FLAG.USER_AUDIO_CALLBACK
        | INIT_FLAG.NO_DEBUG_OUTPUT,
    )
    try:
        slot = Slot(project_path.absolute())
        audio_frames = min(slot.get_song_length_frames(), MAX_FRAMES)
        video_duration = audio_frames // FREQ
        output = np.zeros((audio_frames, 2), DATA_TYPE)
        buffer = np.zeros((audio_frames_per_video_frame, 2), DATA_TYPE)
        output_snapshots = []
        output_50ms_frames = int(FREQ / 1000 * 50)
        click.echo(
            f"Rendering {audio_frames} frames of audio, "
            f"{audio_frames_per_video_frame} frames at a time..."
        )
        position = 0
        with tqdm(total=audio_frames, unit="frame", unit_scale=True) as bar:
            slot.play_from_beginning()
            while position < audio_frames:
                # Grab all master output.
                audio_callback(
                    buffer.ctypes.data_as(CDATA_TYPE),
                    audio_frames_per_video_frame,
                    0,
                    get_ticks(),
                )
                end_pos = min(position + audio_frames_per_video_frame, audio_frames)
                copy_size = end_pos - position
                output[position:end_pos] = buffer[:copy_size]
                position = end_pos

                # Grab Output module snapshots.
                output_snapshot_l = np.zeros((output_50ms_frames,), SCOPE_DATA_TYPE)
                output_snapshot_r = np.zeros((output_50ms_frames,), SCOPE_DATA_TYPE)
                received_l = slot.get_module_scope2(
                    0,
                    0,
                    output_snapshot_l.ctypes.data_as(SCOPE_CDATA_TYPE),
                    output_50ms_frames,
                )
                received_r = slot.get_module_scope2(
                    0,
                    0,
                    output_snapshot_r.ctypes.data_as(SCOPE_CDATA_TYPE),
                    output_50ms_frames,
                )
                assert received_l == received_r
                output_snapshot_l.shape = (received_l,)
                output_snapshot_r.shape = (received_r,)
                output_snapshot = np.block([output_snapshot_l, output_snapshot_r])
                output_snapshots.append(output_snapshot)

                bar.update(copy_size)
    finally:
        deinit()

    click.echo(f"Compositing {video_duration} video frames at {fps} FPS...")
    audio_clip = moviepy.audio.AudioClip.AudioArrayClip(output, FREQ)
    bg_clip = moviepy.editor.ColorClip(
        (width, height),
        color=(0, 0, 0),
        duration=video_duration,
    )
    text_clip = moviepy.editor.TextClip(
        slot.get_song_name(),
        font_size=12,
        font=font,
        color="white",
    )
    text_clip = text_clip.with_position("center")
    text_clip = text_clip.with_duration(video_duration)
    video = moviepy.editor.CompositeVideoClip(
        [bg_clip, text_clip],
        size=(width, height),
    )
    video = video.with_audio(audio_clip)
    video = video.with_duration(video_duration)

    click.echo(f"Writing to {output_path}...")
    video.write_videofile(
        str(output_path),
        fps=fps,
        bitrate=f"{video_bitrate}k",
        codec=video_codec,
        audio_codec=audio_codec,
        audio_fps=FREQ,
        audio_bitrate=f"{audio_bitrate}k",
        remove_temp=False,
    )


if __name__ == "__main__":
    main()
