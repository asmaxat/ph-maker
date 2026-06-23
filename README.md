# Power Hour Maker

Concatenate a folder of videos into a single power hour video with shot counter and beer can animations.

## Requirements

- Python 3.10+
- FFmpeg (with ffprobe)

## Usage

```bash
python3 ph_maker.py /path/to/videos/folder -o output.mp4
```

The script will:
1. Scan the folder for video files (`.mp4`, `.mov`, `.avi`, `.mkv`, `.webm`)
2. Sort them chronologically (oldest first)
3. Concatenate them into a single video (no cutting)
4. Add a shot counter in the bottom-right corner (increments every 60 seconds)
5. Flash a beer can image (bottom-left) for 1 second at each minute mark
6. Play a beer can opening sound at each minute mark

## Video ordering

Videos are sorted by date using this priority:
1. **Filename date** — prefix like `2024-03-15_party.mp4` or `20240315_party.mp4`
2. **Video metadata** — creation_time from file metadata
3. **File modification date** — last resort fallback

To override the order, rename files with a date prefix.

## Audio options

By default each clip's audio is mixed at full level (the beer-can cues are
layered on top without attenuating the video audio).

- `--volume N` — multiply the overall audio by `N` (e.g. `--volume 2.0` doubles it).
- `--loudness` — loudness-normalize the finished video (EBU R128) so quiet
  dialogue and loud action play at one consistent level. This runs a **two-pass**
  `loudnorm` (measure, then apply): the audio is re-encoded but the video is
  copied, so it's fast. Single-pass normalization is avoided because it
  undershoots badly on long, dynamic source material.
- `--lufs TARGET` — integrated loudness target used with `--loudness`
  (default `-14`, roughly streaming level; `-12` is noticeably louder).

```bash
# Loud, consistent power hour at streaming-ish level
python3 ph_maker.py /path/to/videos -o output.mp4 --loudness --lufs -12
```

Note: a target is approached, not guaranteed — material with very wide dynamic
range and hot true peaks may land a dB or two short once peaks are limited.

## Running tests

```bash
python3 -m pytest tests/ -v
```
