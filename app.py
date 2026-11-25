import os, re, json, uuid, shutil, subprocess
from pathlib import Path
from urllib.parse import urlparse

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel, Field
from yt_dlp import YoutubeDL
from openai import OpenAI

APP_DIR = Path(__file__).parent
JOBS_DIR = Path(os.getenv("JOBS_DIR", str(APP_DIR / "jobs")))
JOBS_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Clips Adeline Viral AI")

class CreateRequest(BaseModel):
    url: str
    formats: list[str] = Field(default_factory=lambda: ["vertical", "horizontal"])
    max_clips: int = Field(default=5, ge=1, le=15)

def run(command):
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError((result.stderr or result.stdout or "FFmpeg error")[-2500:])
    return result.stdout

def valid_youtube_url(value):
    parsed = urlparse(value)
    host = (parsed.hostname or "").lower()
    return parsed.scheme == "https" and (
        host == "youtu.be" or host == "youtube.com" or host.endswith(".youtube.com")
    )

def srt_time(seconds):
    ms = max(0, int(round(seconds * 1000)))
    hours, ms = divmod(ms, 3600000)
    minutes, ms = divmod(ms, 60000)
    secs, ms = divmod(ms, 1000)
    return f"{hours:02}:{minutes:02}:{secs:02},{ms:03}"

def score_text(text):
    # Heurística inicial: no es una predicción garantizada de viralidad.
    words = text.lower()
    hooks = ["pero", "nunca", "secreto", "increíble", "nadie", "error",
             "verdad", "por qué", "cómo", "descubrí", "millones",
             "impactante", "problema", "solución", "imagina"]
    score = 42 + sum(5 for hook in hooks if hook in words)
    score += min(len(text.split()) // 8, 18)
    if "?" in text or "!" in text:
        score += 5
    return min(95, max(35, score))

def build_srt(segments, clip_start, clip_end, target):
    lines = []
    idx = 1
    for segment in segments:
        start = max(float(segment["start"]), clip_start)
        end = min(float(segment["end"]), clip_end)
        text = str(segment.get("text", "")).strip()
        if end <= start or not text:
            continue
        lines.extend([
            str(idx),
            f"{srt_time(start-clip_start)} --> {srt_time(end-clip_start)}",
            text,
            ""
        ])
        idx += 1
    target.write_text("\n".join(lines), encoding="utf-8")

@app.get("/", response_class=HTMLResponse)
def home():
    return (APP_DIR / "index.html").read_text(encoding="utf-8")

@app.get("/health")
def health():
    return {"status": "ok", "app": "Clips Adeline Viral AI"}

@app.post("/api/create")
def create_clips(request: CreateRequest):
    if not valid_youtube_url(request.url):
        raise HTTPException(400, "Introduce un enlace HTTPS válido de YouTube.")
    if not request.formats or any(item not in {"vertical", "horizontal"} for item in request.formats):
        raise HTTPException(400, "Selecciona al menos un formato válido.")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(503, "Falta configurar OPENAI_API_KEY en Render. La aplicación todavía no puede transcribir videos.")

    job_id = uuid.uuid4().hex[:12]
    folder = JOBS_DIR / job_id
    folder.mkdir(parents=True, exist_ok=True)

    try:
        options = {
            "format": "bestvideo[height<=1080]+bestaudio/best[height<=1080]",
            "merge_output_format": "mp4",
            "outtmpl": str(folder / "source.%(ext)s"),
            "noplaylist": True,
            "quiet": True,
            "no_warnings": True,
        }
        with YoutubeDL(options) as ydl:
            info = ydl.extract_info(request.url, download=True)
        duration = int(info.get("duration") or 0)
        max_minutes = int(os.getenv("MAX_VIDEO_MINUTES", "90"))
        if duration and duration > max_minutes * 60:
            raise ValueError(f"El video supera el límite de {max_minutes} minutos.")

        source_files = [p for p in folder.glob("source.*") if p.suffix.lower() in {".mp4", ".mkv", ".webm", ".mov"}]
        if not source_files:
            raise RuntimeError("No se encontró el archivo de video descargado.")
        source = source_files[0]
        audio = folder / "audio.mp3"
        run(["ffmpeg", "-y", "-i", str(source), "-vn", "-ac", "1", "-ar", "16000", "-b:a", "64k", str(audio)])

        client = OpenAI(api_key=api_key)
        with audio.open("rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                response_format="verbose_json",
                timestamp_granularities=["segment"],
            )
        segments = [dict(start=s.start, end=s.end, text=s.text) for s in (transcript.segments or [])]
        candidates = []
        for segment in segments:
            text = segment["text"].strip()
            start, end = float(segment["start"]), float(segment["end"])
            if len(text) >= 35 and 4 <= end-start <= 45:
                candidates.append({
                    "start": max(0, start-1.5),
                    "end": min(float(duration or end+1), end+1.5),
                    "text": text,
                    "score": score_text(text),
                })
        # Evita clips casi idénticos y mantiene una separación mínima.
        candidates.sort(key=lambda item: item["score"], reverse=True)
        selected = []
        for candidate in candidates:
            if all(abs(candidate["start"] - existing["start"]) > 35 for existing in selected):
                selected.append(candidate)
            if len(selected) >= request.max_clips:
                break
        if not selected:
            raise RuntimeError("No se encontraron segmentos hablados suficientemente largos para proponer clips.")

        output = []
        selected.sort(key=lambda item: item["start"])
        for number, candidate in enumerate(selected, 1):
            clip_id = f"clip-{number:03d}"
            clip_start, clip_end = candidate["start"], candidate["end"]
            subtitle = folder / f"{clip_id}.srt"
            build_srt(segments, clip_start, clip_end, subtitle)
            formats = {}
            for fmt in request.formats:
                target = folder / f"{clip_id}-{fmt}.mp4"
                if fmt == "vertical":
                    vf = f"scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,subtitles={subtitle}:force_style='FontName=DejaVu Sans,FontSize=18,Outline=2,Alignment=2,MarginV=110'"
                else:
                    vf = f"scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2,subtitles={subtitle}:force_style='FontName=DejaVu Sans,FontSize=24,Outline=2,Alignment=2,MarginV=45'"
                run([
                    "ffmpeg", "-y", "-ss", str(clip_start), "-i", str(source),
                    "-t", str(max(1, clip_end-clip_start)), "-vf", vf,
                    "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
                    "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", str(target)
                ])
                formats[fmt] = f"/files/{job_id}/{target.name}"
            output.append({
                "number": number,
                "score": candidate["score"],
                "text": candidate["text"],
                "formats": formats
            })

        (folder / "results.json").write_text(json.dumps(output, ensure_ascii=False), encoding="utf-8")
        return {"job_id": job_id, "title": info.get("title", "Video de YouTube"), "clips": output}
    except Exception as error:
        shutil.rmtree(folder, ignore_errors=True)
        raise HTTPException(500, f"No se pudo procesar el video: {str(error)[:700]}")

@app.get("/files/{job_id}/{filename}")
def download_file(job_id: str, filename: str):
    if not re.fullmatch(r"[a-f0-9]{12}", job_id) or "/" in filename or "\\" in filename:
        raise HTTPException(404, "Archivo no encontrado.")
    path = (JOBS_DIR / job_id / filename).resolve()
    if not path.is_file() or not path.is_relative_to((JOBS_DIR / job_id).resolve()):
        raise HTTPException(404, "Archivo no encontrado.")
    return FileResponse(path, filename=filename, media_type="video/mp4")
