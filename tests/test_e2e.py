"""End-to-end test that generates real videos with FFmpeg and runs the full pipeline."""

import os
import shutil
import subprocess
import tempfile

import pytest

# Skip the entire module if ffmpeg is not available
pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None,
    reason="FFmpeg not installed",
)


def generate_test_video(path: str, duration: int = 40, color: str = "blue") -> None:
    """Generate a short test video with ffmpeg."""
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", f"color=c={color}:s=320x240:d={duration}",
            "-f", "lavfi", "-i", f"sine=frequency=440:duration={duration}",
            "-c:v", "libx264", "-c:a", "aac",
            "-shortest", path,
        ],
        capture_output=True,
        check=True,
    )


def test_e2e_power_hour():
    with tempfile.TemporaryDirectory() as d:
        # Generate 3 short videos with different durations (~120s total)
        generate_test_video(os.path.join(d, "2024-01-01_red.mp4"), 38, "red")
        generate_test_video(os.path.join(d, "2024-02-01_green.mp4"), 40, "green")
        generate_test_video(os.path.join(d, "2024-03-01_blue.mp4"), 42, "blue")

        output = os.path.join(d, "output.mp4")

        # Import pipeline functions
        from ph_maker import (
            build_ffmpeg_command,
            calculate_shot_times,
            discover_videos,
            get_video_duration,
        )

        # Run the pipeline
        videos = discover_videos(d)
        assert len(videos) == 3

        total = sum(get_video_duration(v) for v in videos)
        shot_times = calculate_shot_times(total)
        assert len(shot_times) >= 1, "Expected at least 1 shot mark for ~120s of video"

        assets_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets"
        )
        cmd = build_ffmpeg_command(videos, shot_times, output, assets_dir)
        result = subprocess.run(cmd, capture_output=True, text=True)
        assert result.returncode == 0, f"FFmpeg failed: {result.stderr}"

        # Verify output exists and has reasonable duration
        assert os.path.exists(output)
        out_duration = get_video_duration(output)
        # Output includes 10s title screen
        expected = total + 10.0
        assert abs(out_duration - expected) < 5.0, (
            f"Output duration {out_duration:.1f}s differs from expected {expected:.1f}s"
        )
