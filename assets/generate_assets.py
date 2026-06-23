#!/usr/bin/env python3
"""Generate bundled assets for Power Hour Maker.

Creates:
  - beercan.png  (200x350, transparent background, beer can graphic)
  - can_open.wav (short percussive pop/click sound)
"""

import subprocess
import sys
from pathlib import Path

ASSETS_DIR = Path(__file__).parent


def generate_beercan_png():
    """Draw a stylized beer can using Pillow."""
    from PIL import Image, ImageDraw, ImageFont

    W, H = 200, 350
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Body dimensions
    body_left, body_right = 30, 170
    body_top, body_bottom = 50, 330
    body_w = body_right - body_left

    # --- Can body (gold/amber gradient effect via stacked rects) ---
    for y in range(body_top, body_bottom):
        t = (y - body_top) / (body_bottom - body_top)
        # Gold-to-amber gradient
        r = int(220 + 35 * t)
        g = int(180 - 40 * t)
        b = int(30 + 20 * t)
        draw.line([(body_left, y), (body_right, y)], fill=(r, g, b, 255))

    # --- Highlight stripe (specular reflection on cylinder) ---
    hl_left = body_left + int(body_w * 0.15)
    hl_right = body_left + int(body_w * 0.30)
    for y in range(body_top + 10, body_bottom - 10):
        t = (y - body_top) / (body_bottom - body_top)
        alpha = int(80 - 30 * abs(t - 0.5))
        draw.line([(hl_left, y), (hl_right, y)], fill=(255, 255, 255, max(alpha, 0)))

    # --- Top rim (ellipse) ---
    rim_h = 16
    draw.ellipse(
        [body_left, body_top - rim_h // 2, body_right, body_top + rim_h // 2],
        fill=(200, 200, 210, 255),
        outline=(140, 140, 150, 255),
    )

    # --- Top surface / lid ---
    lid_inset = 10
    draw.ellipse(
        [
            body_left + lid_inset,
            body_top - rim_h // 2 - 6,
            body_right - lid_inset,
            body_top + rim_h // 2 - 6,
        ],
        fill=(190, 190, 200, 255),
        outline=(150, 150, 160, 255),
    )

    # --- Pull tab ---
    tab_cx = W // 2 + 5
    tab_cy = body_top - 4
    tab_w, tab_h = 22, 12
    draw.ellipse(
        [tab_cx - tab_w // 2, tab_cy - tab_h // 2, tab_cx + tab_w // 2, tab_cy + tab_h // 2],
        fill=(170, 170, 180, 255),
        outline=(120, 120, 130, 255),
        width=2,
    )
    # Tab hole
    draw.ellipse(
        [tab_cx - 4, tab_cy - 3, tab_cx + 4, tab_cy + 3],
        fill=(190, 190, 200, 255),
    )
    # Tab rivet
    draw.ellipse(
        [tab_cx - 12, tab_cy - 2, tab_cx - 8, tab_cy + 2],
        fill=(150, 150, 160, 255),
    )

    # --- Bottom rim ---
    draw.ellipse(
        [body_left, body_bottom - rim_h // 2, body_right, body_bottom + rim_h // 2],
        fill=(180, 170, 100, 255),
        outline=(140, 130, 70, 255),
    )

    # --- Label band ---
    label_top = body_top + 80
    label_bottom = body_bottom - 60
    for y in range(label_top, label_bottom):
        t = (y - label_top) / (label_bottom - label_top)
        r = int(180 + 40 * (1 - abs(2 * t - 1)))
        g = int(40 + 20 * (1 - abs(2 * t - 1)))
        b = 20
        draw.line([(body_left + 2, y), (body_right - 2, y)], fill=(r, g, b, 230))

    # --- "BEER" text ---
    # Try to get a bold font, fall back to default
    font_size = 38
    font = None
    for font_path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
    ]:
        try:
            font = ImageFont.truetype(font_path, font_size)
            break
        except (OSError, IOError):
            continue
    if font is None:
        font = ImageFont.load_default()

    text = "BEER"
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    tx = (body_left + body_right) // 2 - tw // 2
    ty = (label_top + label_bottom) // 2 - th // 2
    # Shadow
    draw.text((tx + 1, ty + 1), text, fill=(80, 20, 0, 200), font=font)
    # Main text
    draw.text((tx, ty), text, fill=(255, 255, 220, 255), font=font)

    # --- Decorative lines on label ---
    line_color = (255, 220, 100, 180)
    draw.line([(body_left + 8, label_top + 6), (body_right - 8, label_top + 6)], fill=line_color, width=2)
    draw.line([(body_left + 8, label_bottom - 6), (body_right - 8, label_bottom - 6)], fill=line_color, width=2)

    out = ASSETS_DIR / "beercan.png"
    img.save(str(out))
    print(f"Created {out}  ({W}x{H})")


def generate_can_open_wav():
    """Synthesize a short pop/crack sound using ffmpeg lavfi filters."""
    out = ASSETS_DIR / "can_open.wav"

    # Use a single filter_complex graph:
    #   - Click: 2kHz sine burst with fast exponential decay
    #   - Thump: 150Hz sine with moderate decay
    #   - Fizz: white noise burst fading out
    # All mixed together for a realistic can-opening pop.
    filtergraph = (
        "sine=frequency=2000:duration=0.5:sample_rate=44100,volume=eval=frame:volume='if(lt(t,0.05),exp(-40*t),0)'[click];"
        "sine=frequency=150:duration=0.5:sample_rate=44100,volume=eval=frame:volume='if(lt(t,0.1),exp(-20*t),0)'[thump];"
        "anoisesrc=color=white:duration=0.5:sample_rate=44100:amplitude=0.3,volume=eval=frame:volume='exp(-8*t)'[fizz];"
        "[click][thump][fizz]amix=inputs=3:normalize=0,volume=3,alimiter=limit=0.9"
    )
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", filtergraph,
        "-ac", "1",
        "-ar", "44100",
        "-t", "0.5",
        str(out),
    ]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    print(f"Created {out}  (0.5s)")


if __name__ == "__main__":
    print("Generating Power Hour Maker assets...")
    generate_beercan_png()
    generate_can_open_wav()
    print("Done.")
