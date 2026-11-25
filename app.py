import os
import re
import json
import uuid
import shutil
import subprocess
from pathlib import Path
from urllib.parse import urlparse

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel, Field
from yt_dlp import YoutubeDL
from faster_whisper import WhisperModel

APP_DIR = Path(__file__).parent
JOBS_DIR = Path(os.getenv("JOBS_DIR", str(APP_DIR / "jobs")))
JOBS_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Clips Adeline Viral AI")

WHISPER_MODEL = None
CHUNK_SECONDS = 300


class CreateRequest(BaseModel):
    url: str
    formats: list[str] = Field(
        default_factory=lambda: ["vertical"]
    )
    max_clips: int = Field(default=3, ge=1, le=15)
    clip_duration: int = Field(default=30, ge=15, le=90)


def run(command):
    result = subprocess.run(
        command,
        capture_output=True,
        text=True
    )
    if result.returncode:
        raise RuntimeError(
            (result.stderr or result.stdout or "Error de FFmpeg")[-2000:]
        )
    return result.stdout


def get_whisper_model():
    global WHISPER_MODEL

    if WHISPER_MODEL is None:
        WHISPER_MODEL = WhisperModel(
            "tiny",
            device="cpu",
            compute_type="int8",
            cpu_threads=2,
            num_workers=1,
            download_root="/tmp/whisper-models"
        )

    return WHISPER_MODEL


def valid_youtube_url(value):
    parsed = urlparse(value)
    host = (parsed.hostname or "").lower()

    return (
        parsed.scheme == "https"
        and (
            host == "youtu.be"
            or host == "youtube.com"
            or host.endswith(".youtube.com")
        )
    )


def srt_time(seconds):
    ms = max(0, int(round(seconds * 1000)))
    hours, ms = divmod(ms, 3600000)
    minutes, ms = divmod(ms, 60000)
    secs, ms = divmod(ms, 1000)

    return f"{hours:02}:{minutes:02}:{secs:02},{ms:03}"


def score_text(text):
    words = text.lower()

    hooks = [
        "pero", "nunca", "secreto", "increíble",
        "nadie", "error", "verdad", "por qué",
        "cómo", "descubrí", "millones",
        "impactante", "problema", "solución",
        "imagina"
    ]

    score = 42 + sum(
        5 for hook in hooks if hook in words
    )

    score += min(len(text.split()) // 8, 18)

    if "?" in text or "!" in text:
        score += 5

    return min(95, max(35, score))


def transcribe_in_chunks(audio, folder):
    model = get_whisper_model()
    chunk_folder = folder / "audio_chunks"
    chunk_folder.mkdir(exist_ok=True)

    pattern = str(chunk_folder / "chunk-%03d.mp3")

    run([
        "ffmpeg", "-y",
        "-i", str(audio),
        "-f", "segment",
        "-segment_time", str(CHUNK_SECONDS),
        "-reset_timestamps", "1",
        "-c:a", "libmp3lame",
        "-b:a", "48k",
        pattern
    ])

    all_segments = []

    chunks = sorted(chunk_folder.glob("chunk-*.mp3"))

    if not chunks:
        raise RuntimeError("No se pudieron dividir los audios.")

    for index, chunk in enumerate(chunks):
        segments, info = model.transcribe(
            str(chunk),
            beam_size=1,
            vad_filter=True
        )

        offset = index * CHUNK_SECONDS

        for segment in segments:
            text = segment.text.strip()

            if not text:
                continue

            all_segments.append({
                "start": float(segment.start) + offset,
                "end": float(segment.end) + offset,
                "text": text
            })

    return all_segments


def build_srt(segments, clip_start, clip_end, target):
    lines = []
    index = 1

    for segment in segments:
        start = max(float(segment["start"]), clip_start)
        end = min(float(segment["end"]), clip_end)
        text = str(segment.get("text", "")).strip()

        if end <= start or not text:
            continue

        lines.extend([
            str(index),
            f"{srt_time(start - clip_start)} --> "
            f"{srt_time(end - clip_start)}",
            text,
            ""
        ])

        index += 1

    target.write_text(
        "\n".join(lines),
        encoding="utf-8"
    )


@app.get("/", response_class=HTMLResponse)
def home():
    return (APP_DIR / "index.html").read_text(
        encoding="utf-8"
    )


@app.get("/health")
def health():
    return {
        "status": "ok",
        "app": "Clips Adeline Viral AI"
    }


@app.post("/api/create")
def create_clips(request: CreateRequest):

    if not valid_youtube_url(request.url):
        raise HTTPException(
            400,
            "Introduce un enlace HTTPS válido de YouTube."
        )

    if not request.formats or any(
        item not in {"vertical", "horizontal"}
        for item in request.formats
    ):
        raise HTTPException(
            400,
            "Selecciona al menos un formato válido."
        )

    if request.clip_duration not in {15, 30, 45, 60, 90}:
        raise HTTPException(
            400,
            "Selecciona una duración válida."
        )

    job_id = uuid.uuid4().hex[:12]
    folder = JOBS_DIR / job_id
    folder.mkdir(parents=True, exist_ok=True)

    try:
        options = {
            "format": "bestvideo[height<=720]+bestaudio/best[height<=720]",
            "merge_output_format": "mp4",
            "outtmpl": str(folder / "source.%(ext)s"),
            "noplaylist": True,
            "quiet": True,
            "no_warnings": True
        }

        with YoutubeDL(options) as ydl:
            info = ydl.extract_info(
                request.url,
                download=True
            )

        duration = int(info.get("duration") or 0)
        max_minutes = int(
            os.getenv("MAX_VIDEO_MINUTES", "15")
        )

        if not duration:
            raise RuntimeError(
                "No se pudo determinar la duración del video."
            )

        if duration > max_minutes * 60:
            raise ValueError(
                f"El video supera el límite de {max_minutes} minutos."
            )

        source_files = [
            p for p in folder.glob("source.*")
            if p.suffix.lower() in {
                ".mp4", ".mkv", ".webm", ".mov"
            }
        ]

        if not source_files:
            raise RuntimeError(
                "No se encontró el video descargado."
            )

        source = source_files[0]
        audio = folder / "audio.mp3"

        run([
            "ffmpeg", "-y",
            "-i", str(source),
            "-vn",
            "-ac", "1",
            "-ar", "16000",
            "-b:a", "48k",
            str(audio)
        ])

        segments = transcribe_in_chunks(
            audio,
            folder
        )

        if not segments:
            raise RuntimeError(
                "No se detectó voz en el video."
            )

        candidates = []

        for segment in segments:
            text = segment["text"].strip()
            start = float(segment["start"])
            end = float(segment["end"])

            if len(text) >= 20 and end > start:
                candidates.append({
                    "start": max(0, start - 1),
                    "end": min(duration, end + 1),
                    "text": text,
                    "score": score_text(text)
                })

        candidates.sort(
            key=lambda item: item["score"],
            reverse=True
        )

        selected = []
        requested_duration = request.clip_duration

        # Limitamos a 3 clips para cuidar los recursos gratuitos.
        max_clips = min(request.max_clips, 3)

        for candidate in candidates:
            center = (
                candidate["start"] + candidate["end"]
            ) / 2

            clip_start = max(
                0,
                center - requested_duration / 2
            )

            clip_end = min(
                duration,
                clip_start + requested_duration
            )

            clip_start = max(
                0,
                clip_end - requested_duration
            )

            if clip_end - clip_start < 10:
                continue

            overlaps = any(
                not (
                    clip_end <= item["start"]
                    or clip_start >= item["end"]
                )
                for item in selected
            )

            if overlaps:
                continue

            selected.append({
                **candidate,
                "start": clip_start,
                "end": clip_end
            })

            if len(selected) >= max_clips:
                break

        if not selected:
            raise RuntimeError(
                "No se encontraron momentos adecuados para crear clips."
            )

        output = []
        selected.sort(key=lambda item: item["start"])

        for number, candidate in enumerate(selected, 1):
            clip_id = f"clip-{number:03d}"
            clip_start = candidate["start"]
            clip_end = candidate["end"]

            subtitle = folder / f"{clip_id}.srt"

            build_srt(
                segments,
                clip_start,
                clip_end,
                subtitle
            )

            formats = {}

            for fmt in request.formats:
                target = folder / f"{clip_id}-{fmt}.mp4"

                if fmt == "vertical":
                    vf = (
                        "scale=720:1280:force_original_aspect_ratio=increase,"
                        "crop=720:1280,"
                        f"subtitles={subtitle}:"
                        "force_style='FontName=DejaVu Sans,"
                        "FontSize=16,Outline=2,Alignment=2,MarginV=70'"
                    )
                else:
                    vf = (
                        "scale=1280:720:force_original_aspect_ratio=decrease,"
                        "pad=1280:720:(ow-iw)/2:(oh-ih)/2,"
                        f"subtitles={subtitle}:"
                        "force_style='FontName=DejaVu Sans,"
                        "FontSize=22,Outline=2,Alignment=2,MarginV=35'"
                    )

                run([
                    "ffmpeg", "-y",
                    "-ss", str(clip_start),
                    "-i", str(source),
                    "-t", str(max(1, clip_end - clip_start)),
                    "-vf", vf,
                    "-c:v", "libx264",
                    "-preset", "ultrafast",
                    "-crf", "28",
                    "-c:a", "aac",
                    "-b:a", "96k",
                    "-movflags", "+faststart",
                    str(target)
                ])

                formats[fmt] = (
                    f"/files/{job_id}/{target.name}"
                )

            output.append({
                "number": number,
                "score": candidate["score"],
                "text": candidate["text"],
                "duration": round(clip_end - clip_start),
                "formats": formats
            })

        (folder / "results.json").write_text(
            json.dumps(output, ensure_ascii=False),
            encoding="utf-8"
        )

        return {
            "job_id": job_id,
            "title": info.get(
                "title",
                "Video de YouTube"
            ),
            "clips": output
        }

    except Exception as error:
        shutil.rmtree(folder, ignore_errors=True)

        raise HTTPException(
            500,
            f"No se pudo procesar el video: {str(error)[:700]}"
        )


@app.get("/files/{job_id}/{filename}")
def download_file(job_id: str, filename: str):

    if (
        not re.fullmatch(r"[a-f0-9]{12}", job_id)
        or "/" in filename
        or "\\" in filename
    ):
        raise HTTPException(
            404,
            "Archivo no encontrado."
        )

    path = (JOBS_DIR / job_id / filename).resolve()
    job_folder = (JOBS_DIR / job_id).resolve()

    if not path.is_file() or not path.is_relative_to(job_folder):
        raise HTTPException(
            404,
            "Archivo no encontrado."
        )

    return FileResponse(
        path,
        filename=filename,
        media_type="video/mp4"
    )
    
