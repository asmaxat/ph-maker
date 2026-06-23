"""Power Hour Maker - concatenate videos with shot counter and beer can overlays."""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}

# Perceptual-hash dedup: a video is fingerprinted by its duration plus an
# average-hash of two sampled frames (10% and 50% in). Two videos are only
# treated as duplicates when the durations are close AND both frame hashes
# match within a small Hamming distance — so re-encodes/copies collapse while
# genuinely different clips of the same length are kept.
PHASH_SAMPLE_POSITIONS = (0.10, 0.50)  # fractions of duration to sample
PHASH_SIZE = 16  # frames downscaled to PHASH_SIZE x PHASH_SIZE grayscale
PHASH_HAMMING_TOL = 12  # bits (out of PHASH_SIZE**2) two frames may differ by

# Memory budget for FFmpeg: ~250 MB per video input for decode + scale + pad buffers,
# plus ~2 GB base overhead. With 328 inputs this hit ~80 GB and froze the machine.
MEMORY_LIMIT_GB = 20
MEMORY_PER_INPUT_MB = 250
MEMORY_OVERHEAD_MB = 2048


def extract_date_from_filename(filename: str) -> datetime | None:
    """Parse date from filename prefix.

    Supports YYYY-MM-DD (e.g. 2024-03-15_party.mp4) and
    YYYYMMDD (e.g. 20240315_party.mp4) formats.
    Returns None if no date found.
    """
    basename = os.path.basename(filename)

    # Try YYYY-MM-DD
    m = re.match(r"(\d{4})-(\d{2})-(\d{2})", basename)
    if m:
        try:
            return datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            pass

    # Try YYYYMMDD
    m = re.match(r"(\d{4})(\d{2})(\d{2})", basename)
    if m:
        try:
            return datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            pass

    return None


def get_metadata_date(filepath: str) -> datetime | None:
    """Extract creation_time from video metadata via ffprobe.

    Returns None if ffprobe is not available or no creation_time found.
    """
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v", "quiet",
                "-print_format", "json",
                "-show_format",
                filepath,
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return None

        data = json.loads(result.stdout)
        creation_time = data.get("format", {}).get("tags", {}).get("creation_time")
        if creation_time:
            # Parse ISO format like "2024-03-15T12:00:00.000000Z"
            return datetime.fromisoformat(creation_time.replace("Z", "+00:00")).replace(tzinfo=None)
    except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError, ValueError):
        return None

    return None


def get_video_date(filepath: str) -> datetime:
    """Get the date for a video file.

    Priority: filename date > metadata date > file modification time.
    Always returns a datetime (mtime is the final fallback).
    """
    # Try filename first
    filename_date = extract_date_from_filename(os.path.basename(filepath))
    if filename_date is not None:
        return filename_date

    # Try metadata
    metadata_date = get_metadata_date(filepath)
    if metadata_date is not None:
        return metadata_date

    # Fall back to file modification time
    mtime = os.path.getmtime(filepath)
    return datetime.fromtimestamp(mtime)


@dataclass(frozen=True)
class VideoFingerprint:
    """Identity of a video for dedup: duration plus per-frame perceptual hashes."""
    duration: float
    hashes: tuple[int, ...]


def average_hash(pixels: bytes) -> int:
    """Average-hash raw grayscale pixels into an integer.

    Bit i is set when pixel i is at or above the mean brightness (LSB = pixel 0).
    Tolerant to re-encoding noise, which only nudges a handful of bits.
    """
    if not pixels:
        return 0
    mean = sum(pixels) / len(pixels)
    h = 0
    for i, value in enumerate(pixels):
        if value >= mean:
            h |= 1 << i
    return h


def hamming_distance(a: int, b: int) -> int:
    """Number of differing bits between two hashes."""
    return bin(a ^ b).count("1")


def fingerprints_match(
    a: VideoFingerprint,
    b: VideoFingerprint,
    duration_tol: float = 0.5,
    phash_tol: int = PHASH_HAMMING_TOL,
) -> bool:
    """Two videos are duplicates when durations are close AND every sampled
    frame hash matches within the Hamming tolerance."""
    if abs(a.duration - b.duration) > duration_tol:
        return False
    if len(a.hashes) != len(b.hashes):
        return False
    return all(
        hamming_distance(ha, hb) <= phash_tol
        for ha, hb in zip(a.hashes, b.hashes)
    )


def deduplicate_by_fingerprint(
    video_paths: list[str],
    fingerprints: dict[str, VideoFingerprint | None],
    sizes: dict[str, int],
    duration_tol: float = 0.5,
    phash_tol: int = PHASH_HAMMING_TOL,
) -> list[str]:
    """Group videos by matching fingerprint, keeping the smallest file in each
    group. Videos without a fingerprint (probe failed) are always kept.
    Pure function — all I/O is done by the caller. Preserves input order.
    """
    probeable = [p for p in video_paths if fingerprints.get(p) is not None]

    groups: list[list[str]] = []
    assigned: set[str] = set()
    for path in probeable:
        if path in assigned:
            continue
        group = [path]
        assigned.add(path)
        for other in probeable:
            if other in assigned:
                continue
            if fingerprints_match(
                fingerprints[path], fingerprints[other], duration_tol, phash_tol
            ):
                group.append(other)
                assigned.add(other)
        groups.append(group)

    kept = {p for p in video_paths if fingerprints.get(p) is None}  # un-probeable
    for group in groups:
        kept.add(min(group, key=lambda p: sizes[p]))

    return [p for p in video_paths if p in kept]


def extract_frame_hash(filepath: str, position: float, duration: float) -> int:
    """Extract one frame at `position` (fraction of duration) and average-hash it.

    Decodes a single frame to a tiny grayscale raw buffer via ffmpeg so we can
    hash it in pure Python without extra dependencies.
    """
    timestamp = max(0.0, position * duration)
    result = subprocess.run(
        [
            "ffmpeg",
            "-v", "quiet",
            "-ss", f"{timestamp:.3f}",
            "-i", filepath,
            "-frames:v", "1",
            "-vf", f"scale={PHASH_SIZE}:{PHASH_SIZE},format=gray",
            "-f", "rawvideo",
            "-",
        ],
        capture_output=True,
        timeout=15,
    )
    if result.returncode != 0 or len(result.stdout) < PHASH_SIZE * PHASH_SIZE:
        raise RuntimeError(f"frame extraction failed for {filepath}")
    return average_hash(result.stdout[: PHASH_SIZE * PHASH_SIZE])


def compute_video_fingerprint(filepath: str) -> VideoFingerprint:
    """Build a fingerprint (duration + sampled frame hashes) for one video."""
    duration = get_video_duration(filepath)
    hashes = tuple(
        extract_frame_hash(filepath, pos, duration)
        for pos in PHASH_SAMPLE_POSITIONS
    )
    return VideoFingerprint(duration=duration, hashes=hashes)


def deduplicate_videos(video_paths: list[str]) -> list[str]:
    """Remove duplicate videos using duration + perceptual frame hashing.

    A video is a duplicate of another when their durations are close and both
    sampled frames look the same — catching re-encodes, copies, and filtered
    variants while keeping genuinely different clips that share a duration.
    Keeps the smallest file from each duplicate group; videos that can't be
    probed are kept as-is. Preserves input order.
    """
    fingerprints: dict[str, VideoFingerprint | None] = {}
    sizes: dict[str, int] = {}
    for path in video_paths:
        sizes[path] = os.path.getsize(path)
        try:
            fingerprints[path] = compute_video_fingerprint(path)
        except (RuntimeError, KeyError, ValueError):
            fingerprints[path] = None  # can't fingerprint — keep as-is

    return deduplicate_by_fingerprint(video_paths, fingerprints, sizes)


def discover_videos(folder: str) -> list[str]:
    """Find all video files in folder, sorted by date (oldest first).

    Returns list of absolute file paths. Deduplication is opt-in via --dedup
    (see deduplicate_videos).
    """
    videos = []
    for entry in os.scandir(folder):
        if entry.is_file():
            _, ext = os.path.splitext(entry.name)
            if ext.lower() in VIDEO_EXTENSIONS:
                videos.append(os.path.abspath(entry.path))

    videos.sort(key=lambda p: get_video_date(p))
    return videos


def get_video_duration(filepath: str) -> float:
    """Get the duration of a video file in seconds via ffprobe."""
    result = subprocess.run(
        [
            "ffprobe",
            "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            filepath,
        ],
        capture_output=True,
        text=True,
        timeout=10,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {filepath}")
    data = json.loads(result.stdout)
    return float(data["format"]["duration"])


def calculate_shot_times(total_duration: float) -> list[float]:
    """Return list of second-marks at each minute boundary within the duration.

    Does not include 0.0 (start) or values >= total_duration.
    """
    times = []
    t = 60.0
    while t < total_duration:
        times.append(t)
        t += 60.0
    return times


def estimate_max_inputs(memory_limit_gb: float = MEMORY_LIMIT_GB) -> int:
    """Max simultaneous FFmpeg video inputs that fit within the memory limit."""
    available_mb = (memory_limit_gb * 1024) - MEMORY_OVERHEAD_MB
    return max(1, int(available_mb / MEMORY_PER_INPUT_MB))


def build_batch_concat_command(
    video_paths: list[str],
    output_path: str,
) -> list[str]:
    """Normalize and concatenate a batch of videos into an intermediate file."""
    n = len(video_paths)
    cmd: list[str] = ["ffmpeg", "-y"]
    for vp in video_paths:
        cmd.extend(["-i", vp])

    filters: list[str] = []
    for i in range(n):
        filters.append(
            f"[{i}:v]scale=1920:1080:force_original_aspect_ratio=decrease,"
            f"pad=1920:1080:(ow-iw)/2:(oh-ih)/2:black,setsar=1[v{i}]"
        )
        filters.append(
            f"[{i}:a]aresample=44100,aformat=sample_fmts=fltp:channel_layouts=stereo[a{i}]"
        )

    concat_inputs = "".join(f"[v{i}][a{i}]" for i in range(n))
    filters.append(f"{concat_inputs}concat=n={n}:v=1:a=1[outv][outa]")

    cmd.extend(["-filter_complex", ";".join(filters)])
    cmd.extend(["-map", "[outv]", "-map", "[outa]"])
    cmd.extend(["-c:v", "libx264", "-preset", "medium", "-crf", "23"])
    cmd.extend(["-c:a", "aac", "-b:a", "192k"])
    cmd.append(output_path)
    return cmd


def build_ffmpeg_command(
    video_paths: list[str],
    shot_times: list[float],
    output_path: str,
    assets_dir: str,
    title_text: str = "POWER HOUR",
    title_duration: float = 10.0,
    volume: float = 1.0,
) -> list[str]:
    """Construct the full FFmpeg command for the power hour video.

    Builds a filter graph that:
    - Prepends a title screen
    - Concatenates all input videos (video + audio)
    - Draws a shot counter (bottom-right, white, font size 96)
    - Overlays beer can image (bottom-left) for ~1s at each minute mark
    - Mixes delayed copies of the can-open sound at each shot time
    - Applies an overall volume gain (``volume``, 1.0 = unchanged)

    Loudness normalization is a separate two-pass step applied afterwards
    (see :func:`normalize_loudness`), because accurate normalization needs to
    measure the finished mix first.
    """
    n_videos = len(video_paths)
    beercan_path = os.path.join(assets_dir, "beercan.png")
    can_sound_path = os.path.join(assets_dir, "can_open.mp3")

    # Get the duration of the can opening sound so the icon matches it
    try:
        can_sound_duration = get_video_duration(can_sound_path)
    except RuntimeError:
        can_sound_duration = 2.0

    # Build input list: all videos, then beercan image, then one can_open.mp3
    cmd: list[str] = ["ffmpeg", "-y"]
    for vp in video_paths:
        cmd.extend(["-i", vp])
    cmd.extend(["-i", beercan_path])
    if shot_times:
        cmd.extend(["-i", can_sound_path])

    # Input indices:
    #   0..n_videos-1          = video files
    #   n_videos               = beercan.png
    #   n_videos+1             = can_open.mp3 (single input, split in filter graph)
    beercan_idx = n_videos
    can_input_idx = n_videos + 1

    # --- Build filter graph ---
    filters: list[str] = []

    # 1. Generate a title screen (black background + "POWER HOUR" text + silent audio)
    title_dur = int(title_duration)
    filters.append(
        f"color=c=black:s=1920x1080:d={title_dur},format=yuv420p,"
        f"drawtext=text='{title_text}':fontsize=120:fontcolor=white:"
        f"x=(w-tw)/2:y=(h-th)/2[titlev]"
    )
    filters.append(
        f"anullsrc=r=44100:cl=stereo,atrim=0:{title_dur},aformat=sample_fmts=fltp[titlea]"
    )

    # 2. Normalize all videos to same resolution (1920x1080) and audio format
    for i in range(n_videos):
        filters.append(
            f"[{i}:v]scale=1920:1080:force_original_aspect_ratio=decrease,"
            f"pad=1920:1080:(ow-iw)/2:(oh-ih)/2:black,setsar=1[v{i}]"
        )
        filters.append(
            f"[{i}:a]aresample=44100,aformat=sample_fmts=fltp:channel_layouts=stereo[a{i}]"
        )

    # 3. Concat title + all videos
    concat_inputs = "[titlev][titlea]" + "".join(f"[v{i}][a{i}]" for i in range(n_videos))
    n_segments = 1 + n_videos  # title + videos
    filters.append(
        f"{concat_inputs}concat=n={n_segments}:v=1:a=1[concatv][concata]"
    )

    # Shift shot times to account for title screen duration
    shifted_shot_times = [st + title_duration for st in shot_times]

    # 4. Drawtext shot counter on concatenated video
    # Counter starts after title screen, shows shot number based on content time
    drawtext = (
        "drawtext="
        rf"text='%{{eif\:floor((t-{title_duration})/60)+1\:d}}':"
        "fontsize=96:fontcolor=white:"
        f"x=w-tw-50:y=h-th-50:enable='gte(t,{title_duration})'"
    )
    filters.append(f"[concatv]{drawtext}[textv]")

    # 3. Scale beer can image to 300px height, maintain aspect ratio
    filters.append(f"[{beercan_idx}:v]scale=-1:300[beerimg]")

    # 6. Overlay beer can image bottom-left, shown for the duration of the sound
    enable_parts = [
        f"between(t,{st:.1f},{st + can_sound_duration:.1f})" for st in shifted_shot_times
    ]
    enable_expr = "+".join(enable_parts) if enable_parts else "0"
    filters.append(
        f"[textv][beerimg]overlay=x=20:y=H-h-20:enable='{enable_expr}'[outv]"
    )

    # 7. Audio: split single can-open input, delay each copy, then amix all
    if shot_times:
        n_shots = len(shot_times)
        split_outputs = "".join(f"[cansplit{i}]" for i in range(n_shots))
        filters.append(f"[{can_input_idx}:a]asplit={n_shots}{split_outputs}")

        delayed_labels: list[str] = []
        for i, st in enumerate(shifted_shot_times):
            delay_ms = int(st * 1000)
            label = f"can{i}"
            filters.append(f"[cansplit{i}]adelay={delay_ms}|{delay_ms}[{label}]")
            delayed_labels.append(f"[{label}]")

        # Mix concatenated audio with all delayed can sounds.
        # normalize=0 keeps each input at full level — the default (normalize=1)
        # divides every input by the stream count, which for a full power hour
        # (~61 streams) would crush the movie audio to near silence.
        n_audio_streams = 1 + n_shots
        mix_inputs = "[concata]" + "".join(delayed_labels)
        filters.append(
            f"{mix_inputs}amix=inputs={n_audio_streams}:duration=longest:normalize=0[mixa]"
        )
    else:
        filters.append("[concata]acopy[mixa]")

    # 8. Apply overall volume gain (1.0 = unchanged). Loudness normalization,
    # if requested, happens in a separate two-pass step after this render.
    filters.append(f"[mixa]volume={volume}[outa]")

    filter_graph = ";".join(filters)

    cmd.extend(["-filter_complex", filter_graph])
    cmd.extend(["-map", "[outv]", "-map", "[outa]"])
    cmd.extend(["-c:v", "libx264", "-preset", "medium", "-crf", "23"])
    cmd.extend(["-c:a", "aac", "-b:a", "192k"])
    cmd.append(output_path)

    return cmd


# Loudness target defaults (EBU R128). TP=-1 dBTP leaves headroom against
# inter-sample peaks; LRA=11 is a typical program loudness range.
DEFAULT_LOUDNESS_LUFS = -14.0
LOUDNESS_TRUE_PEAK = -1.0
LOUDNESS_RANGE = 11.0


def build_loudnorm_measure_command(
    input_path: str,
    target_lufs: float = DEFAULT_LOUDNESS_LUFS,
) -> list[str]:
    """First-pass command: measure the input's loudness as JSON.

    Single-pass loudnorm guesses from a short look-ahead and undershoots badly
    on long, dynamic material, so we measure first and feed the results back
    into the second pass for an accurate hit on the target.
    """
    return [
        "ffmpeg", "-hide_banner", "-nostats", "-vn", "-i", input_path,
        "-af",
        f"loudnorm=I={target_lufs}:TP={LOUDNESS_TRUE_PEAK}:LRA={LOUDNESS_RANGE}:"
        f"print_format=json",
        "-f", "null", "-",
    ]


def parse_loudnorm_json(output: str) -> dict[str, str]:
    """Extract the loudnorm measurement JSON from ffmpeg's stderr output.

    ffmpeg prints the JSON block at the end of the run; pull out the last
    brace-delimited object and parse it. Raises ValueError if absent/invalid.
    """
    start = output.rfind("{")
    end = output.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("no loudnorm JSON found in ffmpeg output")
    return json.loads(output[start : end + 1])


def build_loudnorm_apply_command(
    input_path: str,
    output_path: str,
    measured: dict[str, str],
    target_lufs: float = DEFAULT_LOUDNESS_LUFS,
) -> list[str]:
    """Second-pass command: apply loudnorm using the measured values.

    Copies the video stream untouched and only re-encodes audio, so this is
    fast (no video re-render). Feeding the measured_* values makes loudnorm
    hit the target accurately instead of guessing.
    """
    loudnorm = (
        f"loudnorm=I={target_lufs}:TP={LOUDNESS_TRUE_PEAK}:LRA={LOUDNESS_RANGE}:"
        f"measured_I={measured['input_i']}:"
        f"measured_TP={measured['input_tp']}:"
        f"measured_LRA={measured['input_lra']}:"
        f"measured_thresh={measured['input_thresh']}:"
        f"offset={measured['target_offset']}:"
        f"linear=true"
    )
    return [
        "ffmpeg", "-y", "-hide_banner", "-i", input_path,
        "-map", "0:v:0", "-map", "0:a:0",
        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
        "-af", loudnorm,
        output_path,
    ]


def normalize_loudness(
    input_path: str,
    output_path: str,
    target_lufs: float = DEFAULT_LOUDNESS_LUFS,
) -> None:
    """Two-pass EBU R128 loudness normalization (measure, then apply).

    Raises RuntimeError if either ffmpeg pass fails.
    """
    measure = subprocess.run(
        build_loudnorm_measure_command(input_path, target_lufs),
        capture_output=True, text=True,
    )
    if measure.returncode != 0:
        raise RuntimeError(f"loudnorm measurement failed:\n{measure.stderr}")
    measured = parse_loudnorm_json(measure.stderr)

    apply = subprocess.run(
        build_loudnorm_apply_command(input_path, output_path, measured, target_lufs),
        capture_output=True, text=True,
    )
    if apply.returncode != 0:
        raise RuntimeError(f"loudnorm apply failed:\n{apply.stderr}")


def main():
    parser = argparse.ArgumentParser(
        description="Power Hour Maker - create power hour videos from a folder of clips"
    )
    parser.add_argument("folder", help="Folder containing video files")
    parser.add_argument(
        "-o",
        "--output",
        default="power_hour.mp4",
        help="Output file path (default: power_hour.mp4)",
    )
    parser.add_argument(
        "-t",
        "--title",
        default="POWER HOUR",
        help="Title screen text (default: POWER HOUR)",
    )
    parser.add_argument(
        "--volume",
        type=float,
        default=1.0,
        help="Overall audio volume multiplier (default: 1.0; e.g. 2.0 doubles it)",
    )
    parser.add_argument(
        "--loudness",
        action="store_true",
        help="Loudness-normalize the audio (EBU R128 loudnorm) so quiet "
             "sections and loud action play at one consistent level",
    )
    parser.add_argument(
        "--lufs",
        type=float,
        default=-14.0,
        help="Integrated loudness target in LUFS when --loudness is set "
             "(default: -14, streaming level; -12 is louder)",
    )
    parser.add_argument(
        "--dedup",
        action="store_true",
        help="Remove duplicate videos (matching duration and frame content) "
             "before processing",
    )
    args = parser.parse_args()

    # Resolve assets directory (relative to this script)
    assets_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")

    # 1. Discover and sort videos
    print(f"Scanning {args.folder} for videos...")
    videos = discover_videos(args.folder)
    if not videos:
        print("No video files found!", file=sys.stderr)
        sys.exit(1)
    if args.dedup:
        before = len(videos)
        videos = deduplicate_videos(videos)
        if before != len(videos):
            print(f"Found {before} video files, {before - len(videos)} duplicates removed")
    print(f"Using {len(videos)} videos:")
    for v in videos:
        print(f"  {os.path.basename(v)}")

    # 2. Get total duration
    print("\nCalculating durations...")
    total = 0.0
    for v in videos:
        d = get_video_duration(v)
        total += d
        print(f"  {os.path.basename(v)}: {int(d//60)}:{int(d%60):02d}")
    minutes = int(total // 60)
    seconds = int(total % 60)
    print(f"Total duration: {minutes}:{seconds:02d}")

    # 3. Calculate shot times
    shot_times = calculate_shot_times(total)
    if shot_times:
        print(
            f"\n{len(shot_times)} shot marks at: "
            f"{', '.join(f'{int(t//60)}:{int(t%60):02d}' for t in shot_times)}"
        )
    else:
        print("\nNo shot marks (video is under 1 minute)")

    # 4. Build and run FFmpeg command
    max_inputs = estimate_max_inputs()
    temp_dir = None
    try:
        if len(videos) > max_inputs:
            # Too many videos for a single FFmpeg pass — concat in batches first
            n_batches = (len(videos) + max_inputs - 1) // max_inputs
            print(
                f"\n{len(videos)} videos exceeds safe limit of {max_inputs} "
                f"simultaneous inputs ({MEMORY_LIMIT_GB} GB). "
                f"Processing in {n_batches} batches..."
            )
            temp_dir = tempfile.mkdtemp(prefix="ph_maker_")
            batch_files: list[str] = []
            for batch_num, start in enumerate(range(0, len(videos), max_inputs)):
                batch = videos[start : start + max_inputs]
                temp_path = os.path.join(temp_dir, f"batch_{batch_num:04d}.mp4")
                print(f"  Batch {batch_num + 1}/{n_batches}: {len(batch)} videos...")
                cmd = build_batch_concat_command(batch, temp_path)
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    print(f"FFmpeg batch failed:\n{result.stderr}", file=sys.stderr)
                    sys.exit(1)
                batch_files.append(temp_path)
            videos = batch_files

        # With --loudness we render the mix to an intermediate file first, then
        # loudness-normalize it into the final output (normalization has to
        # measure the finished mix, so it can't be a single filter-graph pass).
        render_output = args.output
        prenorm_path = None
        if args.loudness:
            prenorm_path = args.output + ".prenorm.mp4"
            render_output = prenorm_path

        cmd = build_ffmpeg_command(
            videos, shot_times, render_output, assets_dir,
            title_text=args.title, volume=args.volume,
        )
        print(f"\nRendering to {render_output}...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"FFmpeg failed:\n{result.stderr}", file=sys.stderr)
            sys.exit(1)

        if args.loudness:
            print(f"\nLoudness-normalizing to {args.lufs} LUFS (two-pass)...")
            try:
                normalize_loudness(render_output, args.output, args.lufs)
            except (RuntimeError, ValueError, KeyError) as e:
                print(f"Loudness normalization failed:\n{e}", file=sys.stderr)
                sys.exit(1)
            finally:
                if prenorm_path and os.path.exists(prenorm_path):
                    os.remove(prenorm_path)
    finally:
        if temp_dir:
            shutil.rmtree(temp_dir, ignore_errors=True)

    print(f"Done! Output: {args.output}")


if __name__ == "__main__":
    main()
