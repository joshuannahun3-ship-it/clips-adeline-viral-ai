import os
import re
import json
import uuid
import shutil
import subprocess
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import HTMLResponse, FileResponse
from faster_whisper import WhisperModel

APP_DIR = Path(__file__).parent
JOBS_DIR = Path(os.getenv("JOBS_DIR", str(APP_DIR / "jobs")))
JOBS_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Clips Adeline Viral AI")
WHISPER_MODEL = None
CHUNK_SECONDS = 300
MAX_UPLOAD_BYTES = 100 * 1024 * 1024
MAX_VIDEO_MINUTES = int(os.getenv("MAX_VIDEO_MINUTES", "15"))
ALLOWED_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm", ".m4v"}


def run(command):
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError((result.stderr or result.stdout or "Error de FFmpeg")[-2000:])
    return result.stdout


def get_whisper_model():
    global WHISPER_MODEL
    if WHISPER_MODEL is None:
        WHISPER_MODEL = WhisperModel(
            "tiny", device="cpu", compute_type="int8", cpu_threads=2,
            num_workers=1, download_root="/tmp/whisper-models"
        )
    return WHISPER_MODEL


def get_duration(video_path):
    output = run([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)
    ]).strip()
    try:
        return int(float(output))
    except (TypeError, ValueError):
        raise RuntimeError("No se pudo leer la duración del video.")


def srt_time(seconds):
    ms = max(0, int(round(seconds * 1000)))
    hours, ms = divmod(ms, 3600000)
    minutes, ms = divmod(ms, 60000)
    secs, ms = divmod(ms, 1000)
    return f"{hours:02}:{minutes:02}:{secs:02},{ms:03}"


def score_text(text):
    words = text.lower()
    hooks = [
        "pero", "nunca", "secreto", "increíble", "nadie", "error",
        "verdad", "por qué", "cómo", "descubrí", "millones",
        "impactante", "problema", "solución", "imagina"
    ]
    score = 42 + sum(5 for hook in hooks if hook in words)
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
        "ffmpeg", "-y", "-i", str(audio), "-f", "segment",
        "-segment_time", str(CHUNK_SECONDS), "-reset_timestamps", "1",
        "-c:a", "libmp3lame", "-b:a", "48k", pattern
    ])
    all_segments = []
    chunks = sorted(chunk_folder.glob("chunk-*.mp3"))
    if not chunks:
        raise RuntimeError("No se pudieron dividir los audios.")
    for index, chunk in enumerate(chunks):
        segments, _ = model.transcribe(str(chunk), beam_size=1, vad_filter=True)
        offset = index * CHUNK_SECONDS
        for segment in segments:
            text = segment.text.strip()
            if text:
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
            f"{srt_time(start - clip_start)} --> {srt_time(end - clip_start)}",
            text, ""
        ])
        index += 1
    target.write_text("\n".join(lines), encoding="utf-8")


def create_outputs(source, folder, formats, max_clips, clip_duration, title):
    duration = get_duration(source)
    if duration <= 0:
        raise RuntimeError("El archivo no contiene una duración válida.")
    if duration > MAX_VIDEO_MINUTES * 60:
        raise ValueError(f"El video supera el límite de {MAX_VIDEO_MINUTES} minutos.")

    audio = folder / "audio.mp3"
    run([
        "ffmpeg", "-y", "-i", str(source), "-vn", "-ac", "1", "-ar", "16000",
        "-b:a", "48k", str(audio)
    ])
    segments = transcribe_in_chunks(audio, folder)
    if not segments:
        raise RuntimeError("No se detectó voz en el video.")

    candidates = []
    for segment in segments:
        text = segment["text"].strip()
        start, end = float(segment["start"]), float(segment["end"])
        if len(text) >= 20 and end > start:
            candidates.append({
                "start": max(0, start - 1), "end": min(duration, end + 1),
                "text": text, "score": score_text(text)
            })
    candidates.sort(key=lambda item: item["score"], reverse=True)

    selected = []
    for candidate in candidates:
        center = (candidate["start"] + candidate["end"]) / 2
        clip_start = max(0, center - clip_duration / 2)
        clip_end = min(duration, clip_start + clip_duration)
        clip_start = max(0, clip_end - clip_duration)
        if clip_end - clip_start < 10:
            continue
        overlaps = any(not (clip_end <= item["start"] or clip_start >= item["end"]) for item in selected)
        if overlaps:
            continue
        selected.append({**candidate, "start": clip_start, "end": clip_end})
        if len(selected) >= min(max_clips, 3):
            break
    if not selected:
        raise RuntimeError("No se encontraron momentos adecuados para crear clips.")

    output = []
    selected.sort(key=lambda item: item["start"])
    for number, candidate in enumerate(selected, 1):
        clip_id = f"clip-{number:03d}"
        clip_start, clip_end = candidate["start"], candidate["end"]
        subtitle = folder / f"{clip_id}.srt"
        build_srt(segments, clip_start, clip_end, subtitle)
        output_formats = {}
        for fmt in formats:
            target = folder / f"{clip_id}-{fmt}.mp4"
            if fmt == "vertical":
                vf = (
                    "scale=720:1280:force_original_aspect_ratio=increase,"
                    "crop=720:1280,"
                    f"subtitles={subtitle}:force_style='FontName=DejaVu Sans,"
                    "FontSize=16,Outline=2,Alignment=2,MarginV=70'"
                )
            else:
                vf = (
                    "scale=1280:720:force_original_aspect_ratio=decrease,"
                    "pad=1280:720:(ow-iw)/2:(oh-ih)/2,"
                    f"subtitles={subtitle}:force_style='FontName=DejaVu Sans,"
                    "FontSize=22,Outline=2,Alignment=2,MarginV=35'"
                )
            run([
                "ffmpeg", "-y", "-ss", str(clip_start), "-i", str(source),
                "-t", str(max(1, clip_end - clip_start)), "-vf", vf,
                "-c:v", "libx264", "-preset", "ultrafast", "-crf", "28",
                "-c:a", "aac", "-b:a", "96k", "-movflags", "+faststart", str(target)
            ])
            output_formats[fmt] = f"/files/{folder.name}/{target.name}"
        output.append({
            "number": number, "score": candidate["score"],
            "text": candidate["text"], "duration": round(clip_end - clip_start),
            "formats": output_formats
        })
    (folder / "results.json").write_text(json.dumps(output, ensure_ascii=False), encoding="utf-8")
    return {"job_id": folder.name, "title": title, "clips": output}


@app.get("/", response_class=HTMLResponse)
def home():
    return (APP_DIR / "index.html").read_text(encoding="utf-8")


@app.get("/health")
def health():
    return {"status": "ok", "app": "Clips Adeline Viral AI"}


@app.post("/api/create")
async def create_clips(
    video: UploadFile = File(...),
    formats: str = Form("[\"vertical\"]"),
    max_clips: int = Form(3),
    clip_duration: int = Form(60)
):
    try:
        selected_formats = json.loads(formats)
    except (TypeError, json.JSONDecodeError):
        raise HTTPException(400, "No se pudieron leer los formatos seleccionados.")
    if not isinstance(selected_formats, list) or not selected_formats or any(x not in {"vertical", "horizontal"} for x in selected_formats):
        raise HTTPException(400, "Selecciona al menos un formato válido.")
    if len(set(selected_formats)) != len(selected_formats):
        selected_formats = list(dict.fromkeys(selected_formats))
    if max_clips < 1 or max_clips > 3:
        raise HTTPException(400, "Puedes solicitar entre 1 y 3 clips.")
    if clip_duration not in {15, 30, 45, 60, 90}:
        raise HTTPException(400, "Selecciona una duración válida.")

    original_name = Path(video.filename or "video.mp4").name
    extension = Path(original_name).suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, "Formato no admitido. Usa MP4, MOV, MKV, WEBM o M4V.")

    job_id = uuid.uuid4().hex[:12]
    folder = JOBS_DIR / job_id
    folder.mkdir(parents=True, exist_ok=True)
    source = folder / f"source{extension}"
    total = 0
    try:
        with source.open("wb") as destination:
            while True:
                chunk = await video.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > MAX_UPLOAD_BYTES:
                    raise ValueError("El video pesa más de 100 MB. Reduce su tamaño e inténtalo de nuevo.")
                destination.write(chunk)
        if total == 0:
            raise ValueError("El archivo está vacío. Selecciona un video e inténtalo de nuevo.")
        return create_outputs(
            source, folder, selected_formats, max_clips, clip_duration,
            Path(original_name).stem or "Video subido"
        )
    except HTTPException:
        shutil.rmtree(folder, ignore_errors=True)
        raise
    except Exception as error:
        shutil.rmtree(folder, ignore_errors=True)
        raise HTTPException(500, f"No se pudo procesar el video: {str(error)[:700]}")
    finally:
        await video.close()


@app.get("/files/{job_id}/{filename}")
def download_file(job_id: str, filename: str):
    if not re.fullmatch(r"[a-f0-9]{12}", job_id) or "/" in filename or "\\" in filename:
        raise HTTPException(404, "Archivo no encontrado.")
    path = (JOBS_DIR / job_id / filename).resolve()
    job_folder = (JOBS_DIR / job_id).resolve()
    if not path.is_file() or not path.is_relative_to(job_folder):
        raise HTTPException(404, "Archivo no encontrado.")
    return FileResponse(path, filename=filename, media_type="video/mp4")
        
