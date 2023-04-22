# Must be first import:
import matplotlib.axes  # noqa

import ctypes
from io import BytesIO
from pathlib import Path
from typing import Callable, Iterator, List, Optional, Tuple

import click
import librosa
import librosa.display
import numpy as np
import pkg_resources
from matplotlib import pyplot as plt
from moviepy.audio.AudioClip import AudioArrayClip
from moviepy.editor import ColorClip, CompositeVideoClip, ImageClip, TextClip, VideoClip
from moviepy.video import fx
from sunvox.api import INIT_FLAG, Slot, audio_callback, deinit, get_ticks, init
from tqdm import tqdm

BG_COLOR = (0, 0, 0)

PLAYHEAD_COLOR = (255, 255, 255)

BOTTOM_LOGO_HEIGHT = 24

OutputSnapshotList = List[Tuple[np.ndarray, np.ndarray]]

MAXIMUM_AUDIO_BITRATE = 320
MAXIMUM_VIDEO_BITRATE = 6000

DATA_TYPE = np.float32
CDATA_TYPE = ctypes.POINTER(ctypes.c_float)

SCOPE_DATA_TYPE = np.int16
SCOPE_CDATA_TYPE = ctypes.POINTER(ctypes.c_int16)

SUNVID_ROOT = Path(__file__).parent
DEFAULT_SUNDOGMEDIUM_PATH = SUNVID_ROOT / "SunDogMedium.ttf"
DEFAULT_SUNVOX_LOGO_PATH = SUNVID_ROOT / "sunvox-logo.png"
DEFAULT_SUNVOX_LOGO_TEXT = "Powered by SunVox - https://warmplace.ru/"
DEFAULT_OUTPUT_PATH_TEMPLATE = "{project_path.stem}-{width}x{height}-{fps}fps.mp4"
DEFAULT_SONG_NAME_TEMPLATE = "{song_name}"

PX = 1 / plt.rcParams["figure.dpi"]


@click.group("sunvid")
def main():
    pass


@main.command("version")
def version():
    info = pkg_resources.get_distribution("sunvid")
    click.echo(info)


@main.command("render")
@click.argument("project-path", type=Path)
@click.option("--output-path-template", type=str, default=DEFAULT_OUTPUT_PATH_TEMPLATE)
@click.option("--song-name-template", type=str, default=DEFAULT_SONG_NAME_TEMPLATE)
@click.option("--fps", type=int, default=15)
@click.option("--width", type=int, default=320)
@click.option("--height", type=int, default=320)
@click.option("--font", type=str, default=str(DEFAULT_SUNDOGMEDIUM_PATH))
@click.option("--sunvox-logo-path", type=Path, default=DEFAULT_SUNVOX_LOGO_PATH)
@click.option("--sunvox-logo-text", type=str, default=DEFAULT_SUNVOX_LOGO_TEXT)
@click.option("--audio-bitrate", type=int, default=None)
@click.option("--video-bitrate", type=int, default=None)
@click.option("--audio-codec", type=str, default="aac")
@click.option("--video-codec", type=str, default="libx264")
@click.option("--audio-sample-rate", type=int, default=48000)
@click.option("--overwrite", type=bool, is_flag=True, default=False)
@click.option("--preview", type=bool, is_flag=True, default=False)
@click.option("--max-kb", type=int, default=25000)
def render(
    project_path: Path,
    output_path_template: str,
    song_name_template: str,
    fps: int,
    width: int,
    height: int,
    font: str,
    sunvox_logo_path: Path,
    sunvox_logo_text: str,
    audio_bitrate: Optional[int],
    video_bitrate: Optional[int],
    audio_codec: str,
    video_codec: str,
    audio_sample_rate: int,
    overwrite: bool,
    preview: bool,
    max_kb: int,
):
    project_path = project_path.absolute()
    if not project_path.exists():
        raise FileNotFoundError(f"{project_path} not found")
    output_path = project_path.parent / output_path_template.format(**locals())
    if song_name_template.startswith("@"):
        song_name_template_path = Path(song_name_template[1:])
        song_name_template = song_name_template_path.read_text()
    if not preview and output_path.exists() and not overwrite:
        raise FileExistsError(f"{output_path} already exists")
    audio_frames_per_video_frame = audio_sample_rate // fps
    if audio_sample_rate / fps != audio_frames_per_video_frame:
        raise ValueError(f"{audio_sample_rate} not evenly divisible by {fps}")
    init(
        None,
        audio_sample_rate,
        2,
        INIT_FLAG.AUDIO_FLOAT32
        | INIT_FLAG.ONE_THREAD
        | INIT_FLAG.USER_AUDIO_CALLBACK
        | INIT_FLAG.NO_DEBUG_OUTPUT,
    )
    try:
        slot = Slot(project_path.absolute())
        song_frames = slot.get_song_length_frames()
        if song_frames == 0:
            click.echo("The project has no playable patterns. Exiting.")
            exit(1)

        if not (audio_bitrate and video_bitrate):
            # Calculate maximum bitrates that will fit within the maximum file size.
            min_audio_bitrate = audio_bitrate or 64
            min_video_bitrate = 0 if video_codec is None else (video_bitrate or 32)
            audio_bitrate, video_bitrate = fit_bitrates_to_size(
                min_audio_bitrate=min_audio_bitrate,
                min_video_bitrate=min_video_bitrate,
                song_frames=song_frames,
                audio_sample_rate=audio_sample_rate,
                max_kb=max_kb,
            )

        max_aframes = max_aframes_for_bitrate(
            bitrate=audio_bitrate + video_bitrate,
            audio_sample_rate=audio_sample_rate,
            max_kb=max_kb,
        )

        audio_frames = min(song_frames, max_aframes)
        video_duration = audio_frames / audio_sample_rate + 1.0 / fps

        output = np.zeros((audio_frames, 2), DATA_TYPE)
        output_snapshots = []
        output_50ms_frames = int(audio_sample_rate / 1000 * 50)
        click.echo(
            f"Rendering {audio_frames} frames of audio, "
            f"{audio_frames_per_video_frame} frames at a time, "
            f"at {audio_bitrate}kbps audio, {video_bitrate}kbps video..."
        )

        position = 0
        with tqdm(total=audio_frames, unit="frame", unit_scale=True) as bar:
            for bytes_copied in render_audio_to_buffers(
                slot=slot,
                audio_sample_rate=audio_sample_rate,
                audio_frames=audio_frames,
                audio_frames_per_video_frame=audio_frames_per_video_frame,
                output=output,
                output_snapshots=output_snapshots,
            ):
                bar.update(bytes_copied)
    finally:
        deinit()

    click.echo(f"Compositing {video_duration} seconds of video at {fps} FPS...")

    audio_clip = AudioArrayClip(output, audio_sample_rate)

    bg_clip = ColorClip((width, height), color=BG_COLOR, duration=video_duration)

    text = song_name_template.format(song_name=slot.get_song_name())
    text_clip: TextClip = TextClip(
        "\n" + text,
        font_size=12,
        font=font,
        color="white",
    )
    text_clip = text_clip.with_position(("center", "top"))

    osc_w = width // 3
    osc_h = int(osc_w * (9 / 16))
    osc_y = height - osc_h - BOTTOM_LOGO_HEIGHT

    osc_clip_l = VideoClip(
        make_frame=osc_frame_maker(
            channel=0,
            h=osc_h,
            w=osc_w,
            fps=fps,
            output_snapshots=output_snapshots,
        ),
    ).with_position((0, osc_y))

    osc_clip_r = VideoClip(
        make_frame=osc_frame_maker(
            channel=1,
            h=osc_h,
            w=osc_w,
            fps=fps,
            output_snapshots=output_snapshots,
        ),
    ).with_position((width - osc_w, osc_y))

    xy_clip = VideoClip(
        make_frame=xy_frame_maker(
            h=osc_h,
            w=osc_w,
            fps=fps,
            output_snapshots=output_snapshots,
        )
    ).with_position((osc_w, osc_y))

    sunvox_logo_clip = ImageClip(sunvox_logo_path).with_position(
        (0, height - BOTTOM_LOGO_HEIGHT),
    )

    sunvox_text_clip: TextClip = TextClip(
        sunvox_logo_text,
        font_size=12,
        font=font,
        color="white",
    )
    sunvox_text_clip = sunvox_text_clip.with_position(
        (30, height - (BOTTOM_LOGO_HEIGHT * 0.75)),  # [TODO] math needs work here
    )

    clip_height = int(width * (9 / 40))
    spec_clip, wave_clip = render_spec_and_wave_imageclips(
        stereo_audio=output,
        audio_sample_rate=audio_sample_rate,
        width=width,
        height=clip_height,
    )

    playhead_w = width
    playhead_h = clip_height
    playhead_size = (playhead_w, playhead_h)
    playhead_color_clip = ColorClip(
        playhead_size,
        color=PLAYHEAD_COLOR,
        duration=video_duration,
    )
    playhead_mask = VideoClip(
        make_frame=playhead_frame_maker(playhead_size, video_duration),
        is_mask=True,
    )

    spec_y = osc_y - clip_height
    wave_y = spec_y - clip_height
    spec_clip_playhead = (
        CompositeVideoClip([playhead_color_clip, spec_clip.with_mask(playhead_mask)])
        # .with_mask(playhead_mask)
        .with_position((0, spec_y))
    )
    spec_clip = spec_clip.with_position((0, spec_y))
    wave_clip_playhead = (
        CompositeVideoClip([playhead_color_clip, wave_clip.with_mask(playhead_mask)])
        # .with_mask(playhead_mask)
        .with_position((0, wave_y))
    )
    wave_clip = wave_clip.with_position((0, wave_y))

    video = CompositeVideoClip(
        [
            bg_clip,
            text_clip,
            osc_clip_l,
            osc_clip_r,
            xy_clip,
            sunvox_logo_clip,
            sunvox_text_clip,
            spec_clip,
            spec_clip_playhead,
            wave_clip,
            wave_clip_playhead,
        ],
        size=(width, height),
    )
    video = video.with_audio(audio_clip)
    video = video.with_duration(video_duration)

    if preview:
        click.echo("Previewing...")
        video.preview(fps=fps)
        return

    click.echo(f"Writing to {output_path}...")
    max_bytes = max_kb * 1024
    if video_codec == "none":
        while True:
            audio_clip.write_audiofile(
                str(output_path),
                fps=audio_sample_rate,
                bitrate=f"{audio_bitrate}k",
                codec=audio_codec,
            )
            if (size := output_path.stat().st_size) <= max_bytes:
                break
            if audio_bitrate <= 64:
                break
            audio_bitrate -= 32
            click.echo(
                f"{size} exceeds {max_bytes}; "
                f"rerendering with new audio bitrate {audio_bitrate}"
            )
    else:
        while True:
            video.write_videofile(
                str(output_path),
                fps=fps,
                bitrate=f"{video_bitrate}k",
                codec=video_codec,
                audio_codec=audio_codec,
                audio_fps=audio_sample_rate,
                audio_bitrate=f"{audio_bitrate}k",
                remove_temp=True,
            )
            if (size := output_path.stat().st_size) <= max_bytes:
                break
            if audio_bitrate <= 64 and video_bitrate <= 32:
                break
            if video_bitrate > 32:
                video_bitrate -= 32
                click.echo(
                    f"{size} exceeds {max_bytes}; "
                    f"rerendering with new video bitrate {video_bitrate}"
                )
            else:
                audio_bitrate -= 32
                click.echo(
                    f"{size} exceeds {max_bytes}; "
                    f"rerendering with new audio bitrate {audio_bitrate}"
                )


def max_aframes_for_bitrate(bitrate: int, audio_sample_rate: int, max_kb: int) -> int:
    return int(audio_sample_rate * 60 * max_kb * 8 / bitrate / 60)


def fit_bitrates_to_size(
    min_audio_bitrate: int,
    min_video_bitrate: int,
    song_frames: int,
    audio_sample_rate: int,
    max_kb: int,
) -> Tuple[int, int]:
    audio_bitrate = min_audio_bitrate
    video_bitrate = min_video_bitrate
    while True:
        if audio_bitrate < MAXIMUM_AUDIO_BITRATE:
            new_audio_bitrate = audio_bitrate + 32
            new_video_bitrate = video_bitrate
        elif min_video_bitrate == 0 or video_bitrate >= MAXIMUM_VIDEO_BITRATE:
            break
        else:
            new_audio_bitrate = audio_bitrate
            new_video_bitrate = video_bitrate + 32
        total_bitrate = new_audio_bitrate + new_video_bitrate
        max_aframes = max_aframes_for_bitrate(total_bitrate, audio_sample_rate, max_kb)
        if max_aframes < song_frames:
            break
        audio_bitrate = new_audio_bitrate
        video_bitrate = new_video_bitrate
    return audio_bitrate, video_bitrate


def render_audio_to_buffers(
    slot: Slot,
    audio_sample_rate: int,
    audio_frames: int,
    audio_frames_per_video_frame: int,
    output: np.ndarray,
    output_snapshots: OutputSnapshotList,
) -> Iterator[int]:
    slot.play_from_beginning()
    snapshot_frames = audio_sample_rate // 1000 * 50
    buffer = np.zeros((audio_frames_per_video_frame, 2), DATA_TYPE)
    position = 0
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
        output_snapshot_l = np.zeros((snapshot_frames,), SCOPE_DATA_TYPE)
        output_snapshot_r = np.zeros((snapshot_frames,), SCOPE_DATA_TYPE)
        ref_l = output_snapshot_l.ctypes.data_as(SCOPE_CDATA_TYPE)
        ref_r = output_snapshot_r.ctypes.data_as(SCOPE_CDATA_TYPE)
        received_l = slot.get_module_scope2(0, 0, ref_l, snapshot_frames)
        received_r = slot.get_module_scope2(0, 1, ref_r, snapshot_frames)
        assert received_l == received_r
        output_snapshot_l.shape = (received_l,)
        output_snapshot_r.shape = (received_r,)
        output_snapshots.append((output_snapshot_l, output_snapshot_r))

        # Update progress bar.
        yield copy_size


def osc_frame_maker(
    channel: int,
    h: int,
    w: int,
    fps: int,
    output_snapshots: OutputSnapshotList,
) -> Callable:
    def make_osc_frame(t: float):
        vframe = int(t * fps)
        vframedata = np.zeros((h, w, 3), np.uint8)
        # Draw axis.
        for x in range(w):
            y = h // 2
            vframedata[y][x] = [96, 96, 96]
        if vframe >= len(output_snapshots):
            return vframedata
        snapshots: Tuple[np.ndarray, np.ndarray] = output_snapshots[vframe]
        snapshot = snapshots[channel].astype(np.float32)
        aframes = len(snapshot)
        # Draw scope.
        h2 = h / 2.0
        snapshot = -snapshot
        snapshot /= 32768.0
        snapshot = snapshot.clip(-1.0, 1.0)
        snapshot *= h2
        snapshot += h2
        snapshot = snapshot.astype(np.uint16)
        snapshot = snapshot.clip(0, h - 1)
        for aframe in range(aframes):
            x = int(aframe / aframes * w)
            y = snapshot[aframe]
            vframedata[y][x] = [255, 255, 255]
        return vframedata

    return make_osc_frame


def xy_frame_maker(
    h: int,
    w: int,
    fps: int,
    output_snapshots: OutputSnapshotList,
) -> Callable:
    def make_xy_frame(t: float):
        vframe = int(t * fps)
        vframedata = np.zeros((h, w, 3), np.uint8)
        # Draw axes.
        for x in range(w):
            y = h // 2
            vframedata[y][x] = [96, 96, 96]
        for y in range(h):
            x = w // 2
            vframedata[y][x] = [96, 96, 96]
        if vframe >= len(output_snapshots):
            return vframedata
        snapshots = output_snapshots[vframe]
        xs: np.ndarray = snapshots[0].astype(np.float32)
        ys: np.ndarray = snapshots[1].astype(np.float32)
        aframes = len(xs)
        # Draw scope.
        h2 = h / 2.0
        w2 = w / 2.0
        xs /= 32768.0
        ys /= 32768.0
        xs *= float(h) / w  # Ensure 1:1 pixel placement.
        ys = -ys
        xs *= w2
        xs += w2
        ys *= h2
        ys += h2
        xs = xs.astype(np.int16)
        ys = ys.astype(np.int16)
        for aframe in range(aframes):
            x = xs[aframe]
            y = ys[aframe]
            if x < 0 or x >= w or y < 0 or y >= h:
                continue
            vframedata[y][x] = [255, 255, 255]
        return vframedata

    return make_xy_frame


def playhead_frame_maker(
    size: Tuple[int, int],
    duration: float,
) -> Callable:
    w, h = size

    def make_playhead_frame(t: float):
        x = max(0, min(w - 1, int(w * (t / duration))))
        framedata = np.zeros((h, w), np.uint8)
        for y in range(h):
            if x - 1 >= 0:
                framedata[y][x - 1] = 128
            framedata[y][x] = 255
            if x + 1 < w:
                framedata[y][x + 1] = 128
        return -framedata

    return make_playhead_frame


def render_spec_and_wave_imageclips(
    stereo_audio: np.ndarray,
    audio_sample_rate: int,
    width: int,
    height: int,
    hop_length: int = 1024,
    left_color: str = "white",
    right_color: str = "white",
    alpha: float = 0.5,
) -> Tuple[ImageClip, ImageClip]:
    fig, ax = plt.subplots(nrows=2, ncols=1, sharex="all", squeeze=True)
    spec_ax, wave_ax = ax
    stereo_audio = np.rot90(stereo_audio)
    mono_audio = stereo_audio.mean(0)
    left, right = stereo_audio[0], stereo_audio[1]
    D = librosa.amplitude_to_db(
        np.abs(librosa.stft(mono_audio, hop_length=hop_length)),
        ref=np.max,
    )
    librosa.display.waveshow(
        left,
        sr=audio_sample_rate,
        ax=wave_ax,
        color=left_color,
        alpha=alpha,
    )
    librosa.display.waveshow(
        right,
        sr=audio_sample_rate,
        ax=wave_ax,
        color=right_color,
        alpha=alpha,
    )
    librosa.display.specshow(
        D,
        y_axis="mel",
        sr=audio_sample_rate,
        hop_length=hop_length,
        x_axis="time",
        ax=spec_ax,
    )
    fig.patch.set_visible(False)
    fig.set_frameon(False)
    for a in ax:
        a.axis("off")
        a.set_xmargin(0)
        a.set_ymargin(0)
    fig.tight_layout()
    fig.set_figwidth(width * PX)
    fig.set_figheight(height * 2 * PX)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)
    png_out = BytesIO()
    fig.savefig(png_out)
    png_out.seek(0)
    image_clip = ImageClip(png_out)
    spec_clip = fx.crop(image_clip, 0, 0, -1, height)
    wave_clip = fx.crop(image_clip, 0, height, -1, -1)
    return spec_clip, wave_clip


if __name__ == "__main__":
    main()
