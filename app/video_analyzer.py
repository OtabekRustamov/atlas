"""
YouTube Shorts frame-by-frame visual analyzer.

Downloads YouTube Shorts, extracts frames at regular intervals, and uses
Claude Vision to analyze each frame for people, backgrounds, buildings,
lighting, colors, and style cues — producing JSON + Markdown reports
suitable for recreating similar images/videos with generative AI tools.

Usage:
    export ANTHROPIC_API_KEY=sk-...

    # Single video:
    python app/video_analyzer.py --url "https://www.youtube.com/shorts/O96MiXxXz6k"

    # Multiple videos in one run:
    python app/video_analyzer.py \\
        --url "https://www.youtube.com/shorts/O96MiXxXz6k" \\
        --url "https://www.youtube.com/shorts/6A4Lb1z-yYs" \\
        --url "https://www.youtube.com/shorts/iA6_O5V55bE"

    # From a local file:
    python app/video_analyzer.py --video path/to/video.mp4

    # Control frame density (default 1 fps):
    python app/video_analyzer.py --url "..." --fps 2
"""

import argparse
import base64
import io
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import anthropic
from PIL import Image


# ── helpers ──────────────────────────────────────────────────────────────────

def _video_id(url: str) -> str:
    """Extract the YouTube video ID from a URL for use as a directory name."""
    m = re.search(r"(?:v=|shorts/|youtu\.be/)([A-Za-z0-9_-]{11})", url)
    return m.group(1) if m else re.sub(r"[^\w-]", "_", url)[-20:]


def download_video(url: str, out_dir: Path) -> Path:
    out_path = out_dir / "video.mp4"
    cmd = [
        "yt-dlp",
        "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "--merge-output-format", "mp4",
        "-o", str(out_path),
        url,
    ]
    print(f"[download] {url}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stderr)
        raise RuntimeError("yt-dlp failed — see error above")
    return out_path


def extract_frames(video: Path, out_dir: Path, fps: float = 1.0) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    pattern = str(out_dir / "frame_%04d.jpg")
    cmd = [
        "ffmpeg", "-y", "-i", str(video),
        "-vf", f"fps={fps}",
        "-q:v", "2",
        pattern,
    ]
    print(f"[frames] extracting at {fps} fps …")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stderr)
        raise RuntimeError("ffmpeg failed")
    frames = sorted(out_dir.glob("frame_*.jpg"))
    print(f"[frames] {len(frames)} frames extracted")
    return frames


def image_to_b64(path: Path, max_px: int = 1024) -> tuple[str, str]:
    img = Image.open(path)
    img.thumbnail((max_px, max_px))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    data = base64.standard_b64encode(buf.getvalue()).decode()
    return data, "image/jpeg"


# ── prompts ──────────────────────────────────────────────────────────────────

FRAME_PROMPT = """\
Analyze this video frame in detail for a generative AI reproduction brief.
Return a JSON object with exactly these keys:

{
  "timestamp_hint": "approximate position in video (early/mid/late)",
  "scene_type": "indoor/outdoor/mixed",
  "setting": {
    "description": "concise overall setting",
    "location_cues": ["specific location indicators — country, city, street style, signage language"],
    "architecture": ["building styles, materials, notable structures, era"],
    "background_elements": ["sky, vegetation, streets, vehicles, props, crowds, etc."],
    "lighting": {
      "type": "natural/artificial/mixed",
      "quality": "hard/soft/diffused",
      "direction": "front/back/side/top/rim",
      "color_temp": "warm/neutral/cool",
      "time_of_day": "dawn/morning/midday/afternoon/golden_hour/dusk/night/unknown"
    },
    "color_palette": ["3-6 dominant hex or descriptive color names"],
    "mood": "e.g. vibrant, melancholic, cinematic, energetic, luxurious"
  },
  "people": [
    {
      "count": 1,
      "gender_appearance": "describe apparent presentation neutrally",
      "age_range": "approximate range e.g. 20s-30s",
      "ethnicity_cues": "observed visual cues only, be neutral and specific",
      "clothing": {
        "style": "e.g. streetwear, luxury, casual, athleisure",
        "colors": ["list all visible garment colors"],
        "notable_items": ["specific items: hat brand if visible, bag, shoes, jewelry, etc."]
      },
      "pose": "detailed pose description",
      "expression": "emotion/mood",
      "position_in_frame": "foreground/midground/background + left/center/right",
      "body_language": "confident/relaxed/dynamic/etc."
    }
  ],
  "camera": {
    "shot_type": "close-up/medium/wide/extreme-wide/selfie",
    "angle": "eye-level/low/high/bird-eye/dutch",
    "movement_hint": "static/pan/tilt/handheld/dolly/drone",
    "aspect_ratio": "portrait/landscape/square",
    "style": "cinematic/vlog/documentary/stylized/social-media",
    "depth_of_field": "shallow/deep/unknown"
  },
  "text_graphics": "any visible text, logos, overlays, subtitles (or null)",
  "ai_generation_prompt": "one dense Midjourney/Stable-Diffusion style prompt (60-100 words) that would reproduce this exact frame",
  "key_style_keywords": ["8-12 tags for image/video gen models"]
}

Be specific and factual. Do not invent details not visible in the frame.
Describe clothing colors, architectural details, and background elements precisely.
"""


SUMMARY_PROMPT = """\
You are a generative AI art director. I've analyzed a YouTube Short frame-by-frame.
Here are the per-frame analyses as JSON:

{frames_json}

Based on ALL frames, produce a comprehensive Markdown report with these sections:

## 1. Video Overview
Short summary: genre, mood, narrative arc, estimated location, target audience.

## 2. Location & Setting
Detailed description of all locations seen. Include architecture styles, country/region cues,
time of day, season, street layout, notable landmarks, and what makes the setting visually
distinctive for AI replication.

## 3. People & Characters
All individuals across frames: physical appearance, skin tone, hair, style, full clothing
palette with specific items, accessories, how they move and interact.
Group repeated characters together.

## 4. Visual Style
Camera work, color grading, lighting philosophy, aspect ratio, editing rhythm, cinematography
style. Note any filters, LUTs, or consistent visual treatments.

## 5. Color Story
Dominant palette across the entire video. Include hex codes or precise color names.
Note any color grading, tonal shifts between scenes, skin tone rendering.

## 6. Master Image Generation Prompt
One comprehensive, dense prompt (150-200 words) in Midjourney/Stable Diffusion style
that captures the ENTIRE video's aesthetic. This is the most important section —
make it actionable and specific enough to reproduce a frame from this video.

## 7. Scene-by-Scene Prompts
For each distinct scene/location, a separate compact generation prompt (60-80 words each).
Label each with a scene number and brief title.

## 8. Video Generation Notes
Specific guidance for generating video (not just images): motion style, transitions,
pacing, camera movement patterns, audio/music aesthetic cues.

## 9. Style Tags
Flat list of 20-30 keywords for use in image/video generation models.
Include location tags, style tags, clothing tags, mood tags.

Be concrete and specific. This report will be used directly by a generative AI engineer
to recreate similar content.
"""


CROSS_VIDEO_PROMPT = """\
You are a generative AI art director. I've analyzed multiple YouTube Shorts from the same channel.
Here are the per-video summaries:

{summaries_json}

Produce a cross-video channel analysis Markdown report:

## 1. Channel Identity
What defines this channel visually? Brand aesthetic, recurring themes, target audience.

## 2. Consistent Visual Elements
Colors, locations, clothing styles, camera work that appear across all videos.

## 3. People / Talent Profile
Consistent description of recurring people across all videos.

## 4. Signature Aesthetic
The channel's unique visual signature — what makes it instantly recognizable.

## 5. Universal Generation Prompt
One master prompt (150-200 words) that captures the CHANNEL'S aesthetic rather than
any single video. Use this to generate content that feels native to this channel.

## 6. Per-Video Prompt Summary
Quick-reference table: | Video ID | One-line setting | Key prompt elements |

## 7. Channel Style Tags
30-40 tags covering the channel's consistent style.

## 8. Recommended Generation Settings
Suggested model parameters, style references, negative prompts, and aspect ratios
for generating content similar to this channel.
"""


# ── analysis logic ────────────────────────────────────────────────────────────

def analyze_frame(client: anthropic.Anthropic, frame: Path, idx: int, total: int) -> dict:
    print(f"  [frame {idx+1:03d}/{total}] {frame.name}")
    data, media_type = image_to_b64(frame)
    msg = client.messages.create(
        model="claude-opus-5-5",
        max_tokens=2000,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": data}},
                    {"type": "text", "text": FRAME_PROMPT},
                ],
            }
        ],
    )
    raw = msg.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"raw_text": raw, "parse_error": True}


def generate_summary(client: anthropic.Anthropic, analyses: list[dict], video_id: str) -> str:
    print(f"[summary] generating report for {video_id} …")
    frames_json = json.dumps(analyses, indent=2)
    msg = client.messages.create(
        model="claude-opus-5-5",
        max_tokens=5000,
        messages=[
            {
                "role": "user",
                "content": SUMMARY_PROMPT.format(frames_json=frames_json),
            }
        ],
    )
    return msg.content[0].text


def generate_cross_video_report(client: anthropic.Anthropic, video_summaries: dict[str, str]) -> str:
    print("[cross-video] generating channel analysis …")
    summaries_json = json.dumps(
        {vid_id: summary for vid_id, summary in video_summaries.items()},
        indent=2
    )
    msg = client.messages.create(
        model="claude-opus-5-5",
        max_tokens=6000,
        messages=[
            {
                "role": "user",
                "content": CROSS_VIDEO_PROMPT.format(summaries_json=summaries_json),
            }
        ],
    )
    return msg.content[0].text


# ── per-video pipeline ────────────────────────────────────────────────────────

def analyze_video(
    client: anthropic.Anthropic,
    url: str | None,
    video_path: Path | None,
    out_dir: Path,
    fps: float,
) -> tuple[str, list[dict], str]:
    """Run the full pipeline for one video. Returns (video_id, analyses, summary_md)."""
    out_dir.mkdir(parents=True, exist_ok=True)

    if video_path is None:
        video_path = download_video(url, out_dir)

    vid_id = _video_id(url) if url else video_path.stem
    print(f"\n{'='*60}")
    print(f"Analyzing: {vid_id}")
    print(f"{'='*60}")

    frames_dir = out_dir / "frames"
    frames = extract_frames(video_path, frames_dir, fps=fps)

    analyses = []
    for idx, frame in enumerate(frames):
        result = analyze_frame(client, frame, idx, len(frames))
        result["frame_file"] = frame.name
        analyses.append(result)

    per_frame_path = out_dir / "frames_analysis.json"
    per_frame_path.write_text(json.dumps(analyses, indent=2))
    print(f"[saved] {per_frame_path}")

    summary_md = generate_summary(client, analyses, vid_id)
    report_path = out_dir / "visual_report.md"
    report_path.write_text(summary_md)
    print(f"[saved] {report_path}")

    return vid_id, analyses, summary_md


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Frame-by-frame video visual analyzer")
    parser.add_argument("--url", action="append", dest="urls", metavar="URL",
                        help="YouTube (or other) video URL; repeat for multiple videos")
    parser.add_argument("--video", help="Path to a local video file (single video only)")
    parser.add_argument("--fps", type=float, default=1.0, help="Frames per second to sample (default 1)")
    parser.add_argument("--out", default="output/analysis", help="Output root directory")
    args = parser.parse_args()

    if not args.urls and not args.video:
        parser.error("Provide at least one --url or --video")

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        sys.exit("Set ANTHROPIC_API_KEY env variable first.")

    client = anthropic.Anthropic(api_key=api_key)
    root_out = Path(args.out)
    root_out.mkdir(parents=True, exist_ok=True)

    video_summaries: dict[str, str] = {}

    if args.video:
        # Single local file
        vid_id, _, summary_md = analyze_video(
            client, url=None, video_path=Path(args.video),
            out_dir=root_out / Path(args.video).stem, fps=args.fps
        )
        video_summaries[vid_id] = summary_md
    else:
        # One or more URLs
        for url in args.urls:
            vid_id = _video_id(url)
            vid_out = root_out / vid_id
            vid_id, _, summary_md = analyze_video(
                client, url=url, video_path=None,
                out_dir=vid_out, fps=args.fps
            )
            video_summaries[vid_id] = summary_md

    # Cross-video channel report (only when analyzing 2+ videos)
    if len(video_summaries) >= 2:
        channel_report = generate_cross_video_report(client, video_summaries)
        channel_path = root_out / "channel_analysis.md"
        channel_path.write_text(channel_report)
        print(f"\n[saved] {channel_path}")

    print("\n=== DONE ===")
    print(f"Output root: {root_out}/")
    for vid_id in video_summaries:
        print(f"  {vid_id}/  →  frames_analysis.json + visual_report.md")
    if len(video_summaries) >= 2:
        print(f"  channel_analysis.md  ← cross-video summary")


if __name__ == "__main__":
    main()
