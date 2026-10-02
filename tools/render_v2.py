#!/usr/bin/env python3
"""Render a v2 short-drama episode from an explicit JSON shot manifest.

The source video (and its synchronized dialogue) stays attached to each new
shot. Optional WAV phone/off-screen lines are mixed at *episode* times. Existing
outputs are never overwritten; the MP4 is published only after verification.

Examples:
    python3 tools/render_v2.py examples/episode-demo.json --schema-only
    python3 tools/render_v2.py examples/episode-demo.json
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import imageio_ffmpeg
except ImportError:
    imageio_ffmpeg = None


WIDTH = 720
HEIGHT = 1280
FPS = 24
SAMPLE_RATE = 48000
MAX_CLIPS = 100


class ManifestError(ValueError):
    pass


class RenderError(RuntimeError):
    pass


@dataclass(frozen=True)
class MediaInfo:
    duration: float | None
    has_video: bool
    has_audio: bool
    width: int | None
    height: int | None
    description: str


def ffmpeg_exe() -> str:
    if imageio_ffmpeg is not None:
        return imageio_ffmpeg.get_ffmpeg_exe()
    found = shutil.which("ffmpeg")
    if found:
        return found
    raise RenderError("FFmpeg unavailable; install FFmpeg or imageio-ffmpeg")


def run(cmd: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(cmd, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)
    if check and result.returncode:
        detail = result.stderr.strip()[-4000:] or result.stdout.strip()[-4000:]
        raise RenderError(f"FFmpeg failed (exit {result.returncode}):\n{detail}")
    return result


def probe(path: Path, ffmpeg: str) -> MediaInfo:
    # FFmpeg's input banner is enough for stream presence and duration. Its
    # return code is normally nonzero here because no output was requested.
    result = run([ffmpeg, "-hide_banner", "-i", str(path)], check=False)
    banner = result.stderr + result.stdout
    duration_match = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", banner)
    duration = None
    if duration_match:
        h, m, s = duration_match.groups()
        duration = int(h) * 3600 + int(m) * 60 + float(s)
    video_lines = re.findall(r"Stream #\d+:\d+(?:\[[^]]+\])?(?:\([^)]*\))?: Video:.*", banner)
    audio_lines = re.findall(r"Stream #\d+:\d+(?:\[[^]]+\])?(?:\([^)]*\))?: Audio:.*", banner)
    dimensions = re.search(r"\b(\d{2,5})x(\d{2,5})\b", video_lines[0]) if video_lines else None
    width = int(dimensions.group(1)) if dimensions else None
    height = int(dimensions.group(2)) if dimensions else None
    return MediaInfo(duration, bool(video_lines), bool(audio_lines), width, height, banner)


def finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ManifestError(f"{label} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ManifestError(f"{label} must be finite")
    return result


def resolve_path(manifest: Path, value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ManifestError(f"{label} must be a nonempty path")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = manifest.parent / path
    return path.resolve()


def load_manifest(path: Path, *, schema_only: bool = False) -> dict[str, Any]:
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ManifestError(f"Cannot read manifest {path}: {exc}") from exc
    if not isinstance(doc, dict):
        raise ManifestError("Manifest root must be a JSON object")
    if doc.get("schema_version") != 1:
        raise ManifestError("schema_version must be 1")
    if doc.get("fps", FPS) != FPS or doc.get("width", WIDTH) != WIDTH or doc.get("height", HEIGHT) != HEIGHT:
        raise ManifestError("This renderer requires 720x1280 at 24 fps")
    clips = doc.get("clips")
    if not isinstance(clips, list) or not 1 <= len(clips) <= MAX_CLIPS:
        raise ManifestError(f"clips must contain 1–{MAX_CLIPS} shots")
    frame_counts: list[int] = []
    for index, clip in enumerate(clips, 1):
        label = f"clips[{index - 1}]"
        if not isinstance(clip, dict):
            raise ManifestError(f"{label} must be an object")
        clip["_path"] = resolve_path(path, clip.get("source"), f"{label}.source")
        start = finite_number(clip.get("start"), f"{label}.start")
        end = finite_number(clip.get("end"), f"{label}.end")
        if start < 0 or end <= start:
            raise ManifestError(f"{label} needs 0 <= start < end")
        frames = round((end - start) * FPS)
        if frames < 1:
            raise ManifestError(f"{label} is shorter than one output frame")
        if abs(frames / FPS - (end - start)) > 0.0209:
            raise ManifestError(f"{label} duration cannot be safely rounded to 24 fps")
        frame_counts.append(frames)
        clip["_frames"] = frames
        clip["_duration"] = frames / FPS
        clip["_start"] = start
        clip["_end"] = end
        mode = clip.get("source_audio", "native")
        if mode not in ("native", "ambient", "mute"):
            raise ManifestError(f"{label}.source_audio must be native, ambient, or mute")
        clip["_mode"] = mode
        normalize_audio = clip.get("normalize_audio", mode == "native")
        if not isinstance(normalize_audio, bool):
            raise ManifestError(f"{label}.normalize_audio must be boolean")
        clip["_normalize_audio"] = normalize_audio and mode == "native"
        target_lufs = finite_number(clip.get("target_lufs", -18), f"{label}.target_lufs")
        if not -30 <= target_lufs <= -10:
            raise ManifestError(f"{label}.target_lufs must be between -30 and -10")
        clip["_target_lufs"] = target_lufs
        gain = finite_number(clip.get("audio_gain", 0.35 if mode == "ambient" else 1.0),
                             f"{label}.audio_gain")
        if gain < 0 or gain > 4:
            raise ManifestError(f"{label}.audio_gain must be between 0 and 4")
        clip["_gain"] = 0.0 if mode == "mute" else gain
        crop = clip.get("crop")
        if crop is not None:
            if not isinstance(crop, dict):
                raise ManifestError(f"{label}.crop must be an object")
            if set(crop) != {"x", "y", "width", "height"}:
                raise ManifestError(f"{label}.crop needs exactly x, y, width, height")
            for key in ("x", "y", "width", "height"):
                value = crop[key]
                if isinstance(value, bool) or not isinstance(value, int):
                    raise ManifestError(f"{label}.crop.{key} must be an integer")
                if value < (1 if key in ("width", "height") else 0):
                    raise ManifestError(f"{label}.crop.{key} is out of range")
            clip["_crop"] = crop
        else:
            clip["_crop"] = None
        for fade_name in ("audio_fade_in", "audio_fade_out"):
            fade = finite_number(clip.get(fade_name, 0.06), f"{label}.{fade_name}")
            if fade < 0 or fade > min(1.0, clip["_duration"] / 2):
                raise ManifestError(f"{label}.{fade_name} outside safe range")
            clip[f"_{fade_name}"] = fade
        if not schema_only and not clip["_path"].is_file():
            raise ManifestError(f"Missing source video: {clip['_path']}")

    total = sum(frame_counts) / FPS
    doc["_duration"] = total
    doc["_cuts"] = [sum(frame_counts[:i]) / FPS for i in range(1, len(frame_counts))]
    expected = doc.get("expected_duration")
    if expected is not None and abs(finite_number(expected, "expected_duration") - total) > 1 / FPS:
        raise ManifestError(f"expected_duration is {expected}, but clips total {total:.3f}s")

    doc["_output"] = resolve_path(path, doc.get("output"), "output")
    if doc["_output"].suffix.lower() != ".mp4":
        raise ManifestError("output must be an .mp4 path")

    overlays = doc.get("voice_overs", [])
    if not isinstance(overlays, list):
        raise ManifestError("voice_overs must be an array")
    active_overlays = []
    for index, overlay in enumerate(overlays):
        label = f"voice_overs[{index}]"
        if not isinstance(overlay, dict):
            raise ManifestError(f"{label} must be an object")
        if not isinstance(overlay.get("enabled", True), bool):
            raise ManifestError(f"{label}.enabled must be boolean")
        if not overlay.get("enabled", True):
            continue
        overlay["_path"] = resolve_path(path, overlay.get("file"), f"{label}.file")
        if overlay["_path"].suffix.lower() != ".wav":
            raise ManifestError(f"{label} must point to a WAV file")
        overlay["_start"] = finite_number(overlay.get("start"), f"{label}.start")
        if not 0 <= overlay["_start"] < total:
            raise ManifestError(f"{label}.start outside episode duration")
        overlay["_trim_start"] = finite_number(overlay.get("trim_start", 0), f"{label}.trim_start")
        if overlay["_trim_start"] < 0:
            raise ManifestError(f"{label}.trim_start must be >= 0")
        trim_end = overlay.get("trim_end")
        if trim_end is not None:
            overlay["_trim_end"] = finite_number(trim_end, f"{label}.trim_end")
            if overlay["_trim_end"] <= overlay["_trim_start"]:
                raise ManifestError(f"{label}.trim_end must exceed trim_start")
        else:
            overlay["_trim_end"] = None
        overlay["_gain"] = finite_number(overlay.get("gain", 1), f"{label}.gain")
        if not 0 <= overlay["_gain"] <= 4:
            raise ManifestError(f"{label}.gain must be between 0 and 4")
        overlay["_fade_in"] = finite_number(overlay.get("fade_in", 0.04), f"{label}.fade_in")
        overlay["_fade_out"] = finite_number(overlay.get("fade_out", 0.08), f"{label}.fade_out")
        if min(overlay["_fade_in"], overlay["_fade_out"]) < 0:
            raise ManifestError(f"{label} fades must be >= 0")
        overlay["_effect"] = overlay.get("effect", "none")
        if overlay["_effect"] not in ("none", "telephone"):
            raise ManifestError(f"{label}.effect must be none or telephone")
        if not schema_only and not overlay["_path"].is_file():
            raise ManifestError(f"Missing overlay WAV: {overlay['_path']}")
        active_overlays.append(overlay)
    doc["_voice_overs"] = active_overlays

    subtitle = doc.get("subtitles", {"enabled": False})
    if not isinstance(subtitle, dict):
        raise ManifestError("subtitles must be an object")
    if not isinstance(subtitle.get("enabled", False), bool):
        raise ManifestError("subtitles.enabled must be boolean")
    if subtitle.get("enabled", False):
        subtitle["_path"] = resolve_path(path, subtitle.get("file"), "subtitles.file")
        if subtitle["_path"].suffix.lower() not in (".srt", ".ass"):
            raise ManifestError("subtitles.file must be .srt or .ass")
        if subtitle.get("mode", "burn") not in ("burn", "soft"):
            raise ManifestError("subtitles.mode must be burn or soft")
        if not schema_only and not subtitle["_path"].is_file():
            raise ManifestError(f"Missing subtitles: {subtitle['_path']}")
    doc["_subtitles"] = subtitle
    return doc


def check_sources(doc: dict[str, Any], ffmpeg: str) -> None:
    for index, clip in enumerate(doc["clips"], 1):
        info = probe(clip["_path"], ffmpeg)
        if not info.has_video:
            raise ManifestError(f"Clip {index} has no video: {clip['_path']}")
        if info.duration is not None and clip["_end"] > info.duration + 0.08:
            raise ManifestError(f"Clip {index} ends at {clip['_end']:.2f}s after its {info.duration:.2f}s source")
        crop = clip["_crop"]
        if crop is not None and info.width is not None and info.height is not None:
            if crop["x"] + crop["width"] > info.width or crop["y"] + crop["height"] > info.height:
                raise ManifestError(f"Clip {index} crop exceeds {info.width}x{info.height} source")
        clip["_has_audio"] = info.has_audio
        if clip["_mode"] == "native" and not info.has_audio:
            raise ManifestError(f"Clip {index} requests native dialogue audio but has no audio track: {clip['_path']}")
    for index, overlay in enumerate(doc["_voice_overs"], 1):
        info = probe(overlay["_path"], ffmpeg)
        if not info.has_audio:
            raise ManifestError(f"Overlay {index} has no audio: {overlay['_path']}")
        if info.duration is None:
            raise ManifestError(f"Cannot determine overlay {index} duration")
        trim_end = overlay["_trim_end"] or info.duration
        if trim_end > info.duration + 0.08:
            raise ManifestError(f"Overlay {index} trim_end is after source duration")
        length = trim_end - overlay["_trim_start"]
        if length <= 0:
            raise ManifestError(f"Overlay {index} has no playable audio")
        if overlay["_fade_in"] + overlay["_fade_out"] > length:
            raise ManifestError(f"Overlay {index} fades are longer than its playable audio")
        overlay["_length"] = length


def format_seconds(value: float) -> str:
    return f"{value:.6f}".rstrip("0").rstrip(".") or "0"


def build_command(doc: dict[str, Any], ffmpeg: str, temp_output: Path,
                  subtitle_copy: Path | None) -> list[str]:
    clips = doc["clips"]
    overlays = doc["_voice_overs"]
    cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y"]
    for clip in clips:
        cmd += ["-i", str(clip["_path"])]
    silence_indices: dict[int, int] = {}
    next_input = len(clips)
    for i, clip in enumerate(clips):
        if not clip["_has_audio"] or clip["_mode"] == "mute":
            silence_indices[i] = next_input
            cmd += ["-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo"]
            next_input += 1
    for overlay in overlays:
        overlay["_input_index"] = next_input
        cmd += ["-i", str(overlay["_path"])]
        next_input += 1
    subtitle = doc["_subtitles"]
    soft_sub_index = None
    if subtitle.get("enabled", False) and subtitle.get("mode", "burn") == "soft":
        soft_sub_index = next_input
        cmd += ["-i", str(subtitle["_path"])]

    filters: list[str] = []
    for i, clip in enumerate(clips):
        start, end, frames = clip["_start"], clip["_end"], clip["_frames"]
        duration = clip["_duration"]
        crop = clip["_crop"]
        crop_filter = (f"crop={crop['width']}:{crop['height']}:{crop['x']}:{crop['y']},"
                       if crop else "")
        filters.append(
            f"[{i}:v]trim=start={format_seconds(start)}:end={format_seconds(end)},"
            "setpts=PTS-STARTPTS,"
            f"fps={FPS},{crop_filter}scale={WIDTH}:{HEIGHT}:force_original_aspect_ratio=increase:flags=lanczos,"
            f"crop={WIDTH}:{HEIGHT},setsar=1,format=yuv420p,"
            "tpad=stop_mode=clone:stop_duration=0.25,"
            f"trim=end_frame={frames},setpts=N/({FPS}*TB)[v{i}]"
        )
        if i in silence_indices:
            source = f"[{silence_indices[i]}:a]"
            audio_trim = f"atrim=duration={format_seconds(duration)}"
        else:
            source = f"[{i}:a]"
            audio_trim = f"atrim=start={format_seconds(start)}:end={format_seconds(end)}"
        chain = (
            source + audio_trim + ",asetpts=PTS-STARTPTS,"
            f"aresample={SAMPLE_RATE},aformat=sample_fmts=fltp:channel_layouts=stereo,"
            f"apad=pad_dur=0.25,atrim=duration={format_seconds(duration)},"
            "asetpts=PTS-STARTPTS"
        )
        if clip["_normalize_audio"]:
            chain += (f",loudnorm=I={format_seconds(clip['_target_lufs'])}:TP=-2:LRA=11,"
                      f"aresample={SAMPLE_RATE}")
        chain += f",volume={format_seconds(clip['_gain'])}"
        fade_in = clip["_audio_fade_in"]
        fade_out = clip["_audio_fade_out"]
        if fade_in:
            chain += f",afade=t=in:st=0:d={format_seconds(fade_in)}"
        if fade_out:
            chain += f",afade=t=out:st={format_seconds(duration - fade_out)}:d={format_seconds(fade_out)}"
        filters.append(chain + f"[a{i}]")

    concat_inputs = "".join(f"[v{i}][a{i}]" for i in range(len(clips)))
    filters.append(f"{concat_inputs}concat=n={len(clips)}:v=1:a=1[vbase][abase]")

    if subtitle.get("enabled", False) and subtitle.get("mode", "burn") == "burn":
        assert subtitle_copy is not None
        # Copy the subtitle into the render's temporary directory so its
        # contents and lifetime match this one render.
        filters.append(f"[vbase]subtitles={subtitle_copy}[vout]")
    else:
        filters.append("[vbase]null[vout]")

    audio_labels = ["[abase]"]
    for i, overlay in enumerate(overlays):
        length = overlay["_length"]
        trim_start = overlay["_trim_start"]
        trim_end = overlay["_trim_end"] or trim_start + length
        chain = (
            f"[{overlay['_input_index']}:a]"
            f"atrim=start={format_seconds(trim_start)}:end={format_seconds(trim_end)},"
            f"asetpts=PTS-STARTPTS,aresample={SAMPLE_RATE},"
            "aformat=sample_fmts=fltp:channel_layouts=stereo"
        )
        if overlay["_effect"] == "telephone":
            chain += ",highpass=f=300,lowpass=f=3400"
        chain += f",volume={format_seconds(overlay['_gain'])}"
        if overlay["_fade_in"]:
            chain += f",afade=t=in:st=0:d={format_seconds(overlay['_fade_in'])}"
        if overlay["_fade_out"]:
            chain += (f",afade=t=out:st={format_seconds(length - overlay['_fade_out'])}"
                      f":d={format_seconds(overlay['_fade_out'])}")
        delay_ms = round(overlay["_start"] * 1000)
        chain += f",adelay={delay_ms}:all=1[vo{i}]"
        filters.append(chain)
        audio_labels.append(f"[vo{i}]")
    if overlays:
        filters.append("".join(audio_labels) +
                       f"amix=inputs={len(audio_labels)}:duration=first:normalize=0:dropout_transition=0[mixed]")
    else:
        filters.append("[abase]anull[mixed]")
    filters.append(f"[mixed]alimiter=limit=0.95,atrim=duration={format_seconds(doc['_duration'])}[aout]")

    cmd += ["-filter_complex", ";".join(filters), "-map", "[vout]", "-map", "[aout]"]
    if soft_sub_index is not None:
        cmd += ["-map", f"{soft_sub_index}:s:0", "-c:s", "mov_text", "-metadata:s:s:0", "language=zho"]
    cmd += [
        "-t", format_seconds(doc["_duration"]),
        "-c:v", "libx264", "-preset", "medium", "-crf", "18",
        "-r", str(FPS), "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k", "-ar", str(SAMPLE_RATE),
        "-movflags", "+faststart", str(temp_output),
    ]
    return cmd


def verify_output(path: Path, doc: dict[str, Any], ffmpeg: str,
                  *, strict_cut_check: bool = True) -> None:
    if not path.is_file() or path.stat().st_size < 10000:
        raise RenderError(f"Output missing or too small: {path}")
    info = probe(path, ffmpeg)
    if not info.has_video or not info.has_audio:
        raise RenderError("Rendered MP4 must contain both video and audio")
    if info.duration is None or abs(info.duration - doc["_duration"]) > 0.15:
        raise RenderError(f"Rendered duration {info.duration} differs from planned {doc['_duration']:.3f}s")
    video_line = next((line for line in info.description.splitlines()
                       if "Video:" in line and "Stream #" in line), "")
    if not re.search(r"\b720x1280\b", video_line) or not re.search(r"\b24 fps\b", video_line):
        raise RenderError(f"Rendered video is not 720x1280/24fps: {video_line}")
    decoded = run([ffmpeg, "-v", "error", "-i", str(path), "-map", "0:v:0", "-map", "0:a:0",
                   "-f", "null", "-"], check=False)
    if decoded.returncode:
        raise RenderError(f"MP4 decode check failed:\n{decoded.stderr[-3000:]}")
    if not doc["_cuts"]:
        return
    black = run([ffmpeg, "-hide_banner", "-loglevel", "info", "-i", str(path),
                 "-vf", "blackdetect=d=0.04:pix_th=0.025:pic_th=0.98",
                 "-an", "-f", "null", "-"], check=False)
    if black.returncode:
        raise RenderError(f"Black-frame check failed:\n{black.stderr[-3000:]}")
    intervals = [(float(a), float(b)) for a, b in re.findall(
        r"black_start:([\d.]+)\s+black_end:([\d.]+)", black.stderr)]
    bad_cuts = [cut for cut in doc["_cuts"]
                if any(start <= cut + 0.08 and end >= cut - 0.08 for start, end in intervals)]
    if bad_cuts:
        message = "Near-black frames at internal cuts: " + ", ".join(f"{cut:.2f}s" for cut in bad_cuts)
        if strict_cut_check:
            raise RenderError(message)
        print("warning: " + message, file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="JSON shot manifest")
    parser.add_argument("--output", type=Path, help="new MP4 destination (must not already exist)")
    parser.add_argument("--schema-only", action="store_true",
                        help="check JSON and shot timings without requiring media files")
    parser.add_argument("--validate-only", action="store_true",
                        help="check JSON and source files without rendering")
    parser.add_argument("--warn-black-cuts", action="store_true",
                        help="warn instead of failing if an internal cut contains near-black frames")
    args = parser.parse_args(argv)
    manifest = args.manifest.expanduser().resolve()
    doc = load_manifest(manifest, schema_only=args.schema_only)
    if args.output:
        doc["_output"] = args.output.expanduser().resolve()
    output = doc["_output"]
    print(f"{doc.get('title', manifest.stem)}: {len(doc['clips'])} shots, {doc['_duration']:.3f}s → {output}")
    if args.schema_only:
        return 0
    ffmpeg = ffmpeg_exe()
    check_sources(doc, ffmpeg)
    if args.validate_only:
        print("Source validation passed")
        return 0
    if output.exists():
        raise ManifestError(f"Refusing to overwrite existing output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="render-v2-", dir=str(output.parent)) as tmp:
        tmp_dir = Path(tmp)
        subtitle_copy = None
        subtitle = doc["_subtitles"]
        if subtitle.get("enabled", False) and subtitle.get("mode", "burn") == "burn":
            subtitle_copy = tmp_dir / ("captions" + subtitle["_path"].suffix.lower())
            shutil.copy2(subtitle["_path"], subtitle_copy)
        temp_output = tmp_dir / "episode.mp4"
        cmd = build_command(doc, ffmpeg, temp_output, subtitle_copy)
        run(cmd)
        verify_output(temp_output, doc, ffmpeg, strict_cut_check=not args.warn_black_cuts)
        if output.exists():
            raise ManifestError(f"Output appeared while rendering; refusing to overwrite: {output}")
        temp_output.replace(output)
    print(f"Verified MP4: {output} ({output.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ManifestError, RenderError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
