"""
YouTube Shorts frame-by-frame visual analyzer.

Downloads a YouTube Short, extracts frames at regular intervals, and uses
Claude Vision to analyze each frame for people, backgrounds, buildings,
lighting, colors, and style cues — producing a JSON + Markdown report
suitable for recreating similar images/videos with generative AI tools.

Usage:
    export ANTHROPIC_API_KEY=sk-...
    python app/video_analyzer.py --url "https://www.youtube.com/shorts/O96MiXxXz6k"

    # If you already have the video locally:
    python app/video_analyzer.py --video path/to/video.mp4

    # Control frame density:
    python app/video_analyzer.py --url "..." --fps 2   # 2 frames per second
"""

import argparse
import base64
import json
import os
import subprocess
import sys
from pathlib import Path

import anthropic
from PIL import Image


# ── helpers ──────────────────────────────────────────────────────────────────

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
    import io
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    data = base64.standard_b64encode(buf.getvalue()).decode()
    return data, "image/jpeg"


FRAME_PROMPT = """\
Analyze this video frame in detail for a generative AI reproduction brief.
Return a JSON object with exactly these keys:

{
  "timestamp_hint": "approximate position in video (early/mid/late)",
  "scene_type": "indoor/outdoor/mixed",
  "setting": {
    "description": "concise overall setting",
    "location_cues": ["list of specific location indicators"],
    "architecture": ["building styles, materials, notable structures"],
    "background_elements": ["sky, vegetation, streets, props, etc."],
    "lighting": {
      "type": "natural/artificial/mixed",
      "quality": "hard/soft/diffused",
      "direction": "front/back/side/top",
      "color_temp": "warm/neutral/cool",
      "time_of_day": "dawn/morning/midday/afternoon/golden_hour/dusk/night/unknown"
    },
    "color_palette": ["dominant hex or color names, 3-6 values"],
    "mood": "e.g. vibrant, melancholic, cinematic, energetic"
  },
  "people": [
    {
      "count": 1,
      "gender_appearance": "describe apparent presentation",
      "age_range": "approximate range",
      "ethnicity_cues": "observed visual cues only, be neutral",
      "clothing": {
        "style": "e.g. streetwear, formal, casual",
        "colors": ["list"],
        "notable_items": ["hat, bag, shoes, etc."]
      },
      "pose": "standing/sitting/walking/action description",
      "expression": "emotion/mood",
      "position_in_frame": "foreground/midground/background + left/center/right"
    }
  ],
  "camera": {
    "shot_type": "close-up/medium/wide/extreme-wide",
    "angle": "eye-level/low/high/bird-eye",
    "movement_hint": "static/pan/tilt/handheld/dolly",
    "aspect_ratio": "portrait/landscape/square",
    "style": "cinematic/vlog/documentary/stylized"
  },
  "text_graphics": "any visible text, logos, overlays (or null)",
  "ai_generation_prompt": "one dense Midjourney/Stable-Diffusion style prompt that would reproduce this frame",
  "key_style_keywords": ["5-10 tags for image gen models"]
}

Be specific and factual. Do not invent details not visible in the frame.
"""


def analyze_frame(client: anthropic.Anthropic, frame: Path, idx: int, total: int) -> dict:
    print(f"[analyze] frame {idx+1}/{total}: {frame.name}")
    data, media_type = image_to_b64(frame)
    msg = client.messages.create(
        model="claude-opus-5-5",
        max_tokens=1500,
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
    # strip markdown fences if model wrapped JSON
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"raw_text": raw, "parse_error": True}


SUMMARY_PROMPT = """\
You are a generative AI art director. I've analyzed a YouTube Short frame-by-frame.
Here are the per-frame analyses as JSON:

{frames_json}

Based on ALL frames, produce a comprehensive Markdown report with these sections:

## 1. Video Overview
Short summary: genre, mood, narrative arc.

## 2. Location & Setting
Detailed description of all locations seen. Include architecture styles, country/region cues,
time of day, season, and what makes the setting visually distinctive.

## 3. People & Characters
All individuals across frames: physical appearance, style, clothing palette, how they move
and interact. Group repeated characters together.

## 4. Visual Style
Camera work, color grading, lighting philosophy, aspect ratio, editing rhythm, cinematography style.

## 5. Color Story
Dominant palette across the video. Include hex codes or color names. Note any color grading.

## 6. Master Image Generation Prompt
One comprehensive, dense prompt (120-200 words) in Midjourney/Stable Diffusion style
that captures the entire video's aesthetic. This is the most important section.

## 7. Scene-by-Scene Prompts
For each distinct scene/location, a separate compact generation prompt (50-80 words each).

## 8. Style Tags
Flat list of 15-25 keywords for use in image/video generation models.

Be concrete and specific. This report will be used directly by a generative AI engineer.
"""


def generate_summary(client: anthropic.Anthropic, analyses: list[dict]) -> str:
    print("[summary] generating master report …")
    frames_json = json.dumps(analyses, indent=2)
    msg = client.messages.create(
        model="claude-opus-5-5",
        max_tokens=4000,
        messages=[
            {
                "role": "user",
                "content": SUMMARY_PROMPT.format(frames_json=frames_json),
            }
        ],
    )
    return msg.content[0].text


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Frame-by-frame video visual analyzer")
    parser.add_argument("--url", help="YouTube (or other) video URL")
    parser.add_argument("--video", help="Path to a local video file")
    parser.add_argument("--fps", type=float, default=1.0, help="Frames per second to sample (default 1)")
    parser.add_argument("--out", default="output/analysis", help="Output directory")
    args = parser.parse_args()

    if not args.url and not args.video:
        parser.error("Provide --url or --video")

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        sys.exit("Set ANTHROPIC_API_KEY env variable first.")

    client = anthropic.Anthropic(api_key=api_key)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = out_dir / "frames"

    # 1. Acquire video
    if args.video:
        video_path = Path(args.video)
    else:
        video_path = download_video(args.url, out_dir)

    # 2. Extract frames
    frames = extract_frames(video_path, frames_dir, fps=args.fps)

    # 3. Analyze each frame
    analyses = []
    for idx, frame in enumerate(frames):
        result = analyze_frame(client, frame, idx, len(frames))
        result["frame_file"] = frame.name
        analyses.append(result)

    # 4. Save per-frame JSON
    per_frame_path = out_dir / "frames_analysis.json"
    per_frame_path.write_text(json.dumps(analyses, indent=2))
    print(f"[saved] {per_frame_path}")

    # 5. Generate summary report
    summary_md = generate_summary(client, analyses)
    report_path = out_dir / "visual_report.md"
    report_path.write_text(summary_md)
    print(f"[saved] {report_path}")

    print("\n=== DONE ===")
    print(f"Frames:       {frames_dir}")
    print(f"Per-frame JSON: {per_frame_path}")
    print(f"Master report:  {report_path}")


if __name__ == "__main__":
    main()
