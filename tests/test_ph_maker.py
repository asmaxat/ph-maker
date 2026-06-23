"""Tests for Power Hour Maker."""

import os
import tempfile
import pathlib
from unittest.mock import patch
from datetime import datetime

from ph_maker import (
    extract_date_from_filename,
    get_video_date,
    discover_videos,
    calculate_shot_times,
    build_ffmpeg_command,
    build_loudnorm_measure_command,
    build_loudnorm_apply_command,
    parse_loudnorm_json,
    average_hash,
    hamming_distance,
    fingerprints_match,
    deduplicate_by_fingerprint,
    VideoFingerprint,
)


def test_extract_date_yyyy_mm_dd():
    assert extract_date_from_filename("2024-03-15_party.mp4").year == 2024
    assert extract_date_from_filename("2024-03-15_party.mp4").month == 3
    assert extract_date_from_filename("2024-03-15_party.mp4").day == 15


def test_extract_date_yyyymmdd():
    assert extract_date_from_filename("20240315_party.mp4").year == 2024
    assert extract_date_from_filename("20240315_party.mp4").month == 3
    assert extract_date_from_filename("20240315_party.mp4").day == 15


def test_extract_date_no_date():
    assert extract_date_from_filename("party.mp4") is None


def test_discover_videos_sorted():
    with tempfile.TemporaryDirectory() as d:
        for name in ["2024-06-01_c.mp4", "2024-01-01_a.mp4", "2024-03-15_b.mp4"]:
            pathlib.Path(d, name).write_bytes(b"\x00")
        videos = discover_videos(d)
        stems = [os.path.basename(v) for v in videos]
        assert stems == ["2024-01-01_a.mp4", "2024-03-15_b.mp4", "2024-06-01_c.mp4"]


def test_discover_videos_ignores_non_video():
    with tempfile.TemporaryDirectory() as d:
        pathlib.Path(d, "2024-01-01_a.mp4").write_bytes(b"\x00")
        pathlib.Path(d, "readme.txt").write_bytes(b"\x00")
        pathlib.Path(d, "photo.jpg").write_bytes(b"\x00")
        videos = discover_videos(d)
        assert len(videos) == 1
        assert os.path.basename(videos[0]) == "2024-01-01_a.mp4"


def test_discover_videos_returns_absolute_paths():
    with tempfile.TemporaryDirectory() as d:
        pathlib.Path(d, "2024-01-01_a.mp4").write_bytes(b"\x00")
        videos = discover_videos(d)
        assert all(os.path.isabs(v) for v in videos)


def test_get_video_date_prefers_filename():
    """Filename date should take priority over mtime."""
    with tempfile.TemporaryDirectory() as d:
        p = pathlib.Path(d, "2024-03-15_party.mp4")
        p.write_bytes(b"\x00")
        dt = get_video_date(str(p))
        assert dt.year == 2024
        assert dt.month == 3
        assert dt.day == 15


def test_get_video_date_falls_back_to_mtime():
    """When no date in filename, should fall back to file mtime."""
    with tempfile.TemporaryDirectory() as d:
        p = pathlib.Path(d, "party.mp4")
        p.write_bytes(b"\x00")
        with patch("ph_maker.get_metadata_date", return_value=None):
            dt = get_video_date(str(p))
        assert isinstance(dt, datetime)


def test_discover_videos_all_extensions():
    with tempfile.TemporaryDirectory() as d:
        for ext in [".mp4", ".mov", ".avi", ".mkv", ".webm"]:
            pathlib.Path(d, f"2024-01-01_v{ext}").write_bytes(b"\x00")
        videos = discover_videos(d)
        assert len(videos) == 5


def test_calculate_shot_times():
    times = calculate_shot_times(220.0)
    assert times == [60.0, 120.0, 180.0]


def test_calculate_shot_times_exact_minute():
    times = calculate_shot_times(120.0)
    assert times == [60.0]  # 120.0 is the end, not a shot mark


def test_calculate_shot_times_under_one_minute():
    times = calculate_shot_times(45.0)
    assert times == []


def test_build_ffmpeg_command_structure():
    cmd = build_ffmpeg_command(
        video_paths=["/tmp/a.mp4", "/tmp/b.mp4"],
        shot_times=[60.0, 120.0],
        output_path="/tmp/out.mp4",
        assets_dir="/assets",
    )
    assert isinstance(cmd, list)
    assert cmd[0] == "ffmpeg"
    assert "/tmp/a.mp4" in cmd
    assert "/tmp/b.mp4" in cmd
    assert cmd[-1] == "/tmp/out.mp4"
    cmd_str = " ".join(cmd)
    assert "drawtext" in cmd_str
    assert "overlay" in cmd_str


def test_build_ffmpeg_command_inputs():
    """All video files and asset files appear as inputs."""
    cmd = build_ffmpeg_command(
        video_paths=["/tmp/a.mp4", "/tmp/b.mp4", "/tmp/c.mp4"],
        shot_times=[60.0, 120.0],
        output_path="/tmp/out.mp4",
        assets_dir="/myassets",
    )
    assert "-i" in cmd
    # All videos should be inputs
    for v in ["/tmp/a.mp4", "/tmp/b.mp4", "/tmp/c.mp4"]:
        assert v in cmd
    # Assets should be inputs
    cmd_str = " ".join(cmd)
    assert "/myassets/beercan.png" in cmd_str
    assert "/myassets/can_open.mp3" in cmd_str


def test_build_ffmpeg_command_concat_count():
    """Concat filter should reference the correct number of videos."""
    cmd = build_ffmpeg_command(
        video_paths=["/tmp/a.mp4", "/tmp/b.mp4", "/tmp/c.mp4"],
        shot_times=[60.0, 120.0],
        output_path="/tmp/out.mp4",
        assets_dir="/assets",
    )
    cmd_str = " ".join(cmd)
    # 3 videos + 1 title screen = 4 segments
    assert "n=4:v=1:a=1" in cmd_str


def test_build_ffmpeg_command_overlay_enable():
    """Overlay enable expression should contain between() for each shot time."""
    cmd = build_ffmpeg_command(
        video_paths=["/tmp/a.mp4"],
        shot_times=[60.0, 120.0],
        output_path="/tmp/out.mp4",
        assets_dir="/assets",
    )
    cmd_str = " ".join(cmd)
    # Shot times are shifted by title_duration (10s default), icon shows for sound duration
    assert "between(t,70.0," in cmd_str
    assert "between(t,130.0," in cmd_str


def test_build_ffmpeg_command_audio_delay():
    """Each shot time should produce an adelay filter."""
    cmd = build_ffmpeg_command(
        video_paths=["/tmp/a.mp4"],
        shot_times=[60.0, 120.0],
        output_path="/tmp/out.mp4",
        assets_dir="/assets",
    )
    cmd_str = " ".join(cmd)
    assert "adelay" in cmd_str
    assert "amix" in cmd_str


def test_build_ffmpeg_command_amix_no_normalize():
    """amix must disable normalization so the movie audio isn't crushed by the
    per-input division across ~61 streams in a full power hour."""
    cmd = build_ffmpeg_command(
        video_paths=["/tmp/a.mp4"],
        shot_times=[60.0, 120.0],
        output_path="/tmp/out.mp4",
        assets_dir="/assets",
    )
    cmd_str = " ".join(cmd)
    assert "normalize=0" in cmd_str


def test_build_ffmpeg_command_volume_default_is_unity():
    """Default volume is 1.0 (unchanged)."""
    cmd = build_ffmpeg_command(
        video_paths=["/tmp/a.mp4"],
        shot_times=[60.0, 120.0],
        output_path="/tmp/out.mp4",
        assets_dir="/assets",
    )
    cmd_str = " ".join(cmd)
    assert "volume=1.0" in cmd_str


def test_build_ffmpeg_command_volume_applied():
    """A custom volume multiplier appears as a volume filter feeding [outa]."""
    cmd = build_ffmpeg_command(
        video_paths=["/tmp/a.mp4"],
        shot_times=[60.0, 120.0],
        output_path="/tmp/out.mp4",
        assets_dir="/assets",
        volume=2.0,
    )
    cmd_str = " ".join(cmd)
    assert "volume=2.0[outa]" in cmd_str


def test_build_ffmpeg_command_volume_without_shots():
    """Volume still applies when there are no shot marks (short video)."""
    cmd = build_ffmpeg_command(
        video_paths=["/tmp/a.mp4"],
        shot_times=[],
        output_path="/tmp/out.mp4",
        assets_dir="/assets",
        volume=1.5,
    )
    cmd_str = " ".join(cmd)
    assert "volume=1.5[outa]" in cmd_str


def test_build_ffmpeg_command_has_no_loudnorm():
    """The render command never embeds loudnorm — it's a separate two-pass step."""
    cmd = build_ffmpeg_command(
        video_paths=["/tmp/a.mp4"],
        shot_times=[60.0, 120.0],
        output_path="/tmp/out.mp4",
        assets_dir="/assets",
    )
    assert "loudnorm" not in " ".join(cmd)


# --- Two-pass loudness normalization ---


def test_loudnorm_measure_command_is_analysis_only():
    """Pass 1 reads JSON measurements without writing a real output file."""
    cmd = build_loudnorm_measure_command("/tmp/in.mp4", target_lufs=-12.0)
    cmd_str = " ".join(cmd)
    assert "/tmp/in.mp4" in cmd
    assert "print_format=json" in cmd_str
    assert "I=-12.0" in cmd_str
    # Analysis pass discards output
    assert "-f" in cmd and "null" in cmd


def test_loudnorm_measure_command_default_target():
    """Default measurement target is -14 LUFS."""
    cmd = build_loudnorm_measure_command("/tmp/in.mp4")
    assert "I=-14.0" in " ".join(cmd)


def test_parse_loudnorm_json_extracts_trailing_object():
    """The JSON block is pulled from noisy ffmpeg stderr output."""
    stderr = (
        "ffmpeg version ...\n"
        "[Parsed_loudnorm_0 @ 0x...] \n"
        '{\n'
        '\t"input_i" : "-21.17",\n'
        '\t"input_tp" : "5.55",\n'
        '\t"input_lra" : "22.90",\n'
        '\t"input_thresh" : "-34.21",\n'
        '\t"target_offset" : "2.51"\n'
        '}\n'
    )
    measured = parse_loudnorm_json(stderr)
    assert measured["input_i"] == "-21.17"
    assert measured["target_offset"] == "2.51"


def test_parse_loudnorm_json_raises_without_json():
    """Missing JSON is a hard error, not a silent default."""
    import pytest
    with pytest.raises(ValueError):
        parse_loudnorm_json("no json here")


def test_loudnorm_apply_command_uses_measured_values_and_copies_video():
    """Pass 2 feeds measured_* values and re-encodes audio only."""
    measured = {
        "input_i": "-21.17",
        "input_tp": "5.55",
        "input_lra": "22.90",
        "input_thresh": "-34.21",
        "target_offset": "2.51",
    }
    cmd = build_loudnorm_apply_command(
        "/tmp/in.mp4", "/tmp/out.mp4", measured, target_lufs=-12.0
    )
    cmd_str = " ".join(cmd)
    assert cmd[-1] == "/tmp/out.mp4"
    assert "measured_I=-21.17" in cmd_str
    assert "measured_TP=5.55" in cmd_str
    assert "measured_thresh=-34.21" in cmd_str
    assert "offset=2.51" in cmd_str
    assert "I=-12.0" in cmd_str
    # Video copied (fast), audio re-encoded
    assert "copy" in cmd
    assert "aac" in cmd


# --- Perceptual-hash deduplication ---


def test_average_hash_sets_bits_for_above_mean_pixels():
    """Bit i is set when pixel i is >= the mean brightness (LSB = pixel 0)."""
    # mean of [0, 0, 255, 255] is 127.5; pixels 2 and 3 are above it
    assert average_hash(bytes([0, 0, 255, 255])) == 0b1100


def test_average_hash_uniform_image():
    """A flat image (every pixel == mean) sets every bit."""
    assert average_hash(bytes([100, 100, 100, 100])) == 0b1111


def test_hamming_distance_counts_differing_bits():
    assert hamming_distance(0b1100, 0b1010) == 2
    assert hamming_distance(0b1111, 0b1111) == 0
    assert hamming_distance(0, 0b1111) == 4


def test_fingerprints_match_identical():
    fp = VideoFingerprint(duration=10.0, hashes=(0b1100, 0b0011))
    assert fingerprints_match(fp, fp) is True


def test_fingerprints_differ_when_duration_too_far():
    a = VideoFingerprint(duration=10.0, hashes=(0b1100, 0b0011))
    b = VideoFingerprint(duration=12.0, hashes=(0b1100, 0b0011))
    assert fingerprints_match(a, b, duration_tol=0.5) is False


def test_fingerprints_match_within_phash_tolerance():
    """Re-encodes shift a few bits; within tolerance they still match."""
    a = VideoFingerprint(duration=10.0, hashes=(0b1100, 0b0011))
    b = VideoFingerprint(duration=10.0, hashes=(0b1101, 0b0011))  # 1 bit off
    assert fingerprints_match(a, b, phash_tol=2) is True


def test_fingerprints_differ_when_frame_too_different():
    a = VideoFingerprint(duration=10.0, hashes=(0b0000, 0b0000))
    b = VideoFingerprint(duration=10.0, hashes=(0b1111, 0b0000))  # 4 bits off
    assert fingerprints_match(a, b, phash_tol=2) is False


def test_dedup_keeps_smallest_of_matching_group():
    paths = ["/orig.mp4", "/reencode.mp4"]
    fps = {
        "/orig.mp4": VideoFingerprint(10.0, (0b1100, 0b0011)),
        "/reencode.mp4": VideoFingerprint(10.0, (0b1100, 0b0011)),
    }
    sizes = {"/orig.mp4": 1000, "/reencode.mp4": 5000}
    result = deduplicate_by_fingerprint(paths, fps, sizes)
    assert result == ["/orig.mp4"]


def test_dedup_keeps_distinct_same_length_clips():
    """Two different clips of equal duration must both survive."""
    paths = ["/a.mp4", "/b.mp4"]
    # Distinct frames differ by far more than the Hamming tolerance (12 bits).
    fps = {
        "/a.mp4": VideoFingerprint(10.0, (0, 0)),
        "/b.mp4": VideoFingerprint(10.0, ((1 << 30) - 1, (1 << 30) - 1)),  # 30 bits
    }
    sizes = {"/a.mp4": 1000, "/b.mp4": 1000}
    result = deduplicate_by_fingerprint(paths, fps, sizes)
    assert result == ["/a.mp4", "/b.mp4"]


def test_dedup_keeps_unprobeable_videos():
    """Videos with no fingerprint (probe failed) are always kept."""
    paths = ["/good.mp4", "/broken.mp4"]
    fps = {
        "/good.mp4": VideoFingerprint(10.0, (0b1100, 0b0011)),
        "/broken.mp4": None,
    }
    sizes = {"/good.mp4": 1000, "/broken.mp4": 2000}
    result = deduplicate_by_fingerprint(paths, fps, sizes)
    assert result == ["/good.mp4", "/broken.mp4"]


def test_dedup_preserves_input_order():
    paths = ["/c.mp4", "/a.mp4", "/b.mp4"]
    fps = {
        "/c.mp4": VideoFingerprint(10.0, (0b0001, 0b0000)),
        "/a.mp4": VideoFingerprint(20.0, (0b0010, 0b0000)),
        "/b.mp4": VideoFingerprint(30.0, (0b0100, 0b0000)),
    }
    sizes = {"/c.mp4": 1, "/a.mp4": 1, "/b.mp4": 1}
    result = deduplicate_by_fingerprint(paths, fps, sizes)
    assert result == ["/c.mp4", "/a.mp4", "/b.mp4"]
