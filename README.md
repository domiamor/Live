FROM python:3.10

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
fastapi
uvicorn
yt-dlp
openai-whisper
torch
numpy
pydantic
python-multipart
FROM python:3.11-slim

RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
import os
import uuid
import subprocess
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import yt_dlp
import whisper

app = FastAPI(title="Live Clipper Pro API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOWNLOADS_DIR = os.path.join(BASE_DIR, "downloads")
os.makedirs(DOWNLOADS_DIR, exist_ok=True)

app.mount("/downloads", StaticFiles(directory=DOWNLOADS_DIR), name="downloads")

CLIPS: List[Dict[str, Any]] = []
PROCESSING = False
LAST_MESSAGE = ""

class StreamRequest(BaseModel):
    url: str

def is_valid_youtube_url(url: str) -> bool:
    u = url.strip().lower()
    return "youtube.com" in u or "youtu.be" in u

def download_youtube_video(url: str) -> str:
    video_id = uuid.uuid4().hex[:8]
    output_path = os.path.join(DOWNLOADS_DIR, f"video_{video_id}.%(ext)s")

    ydl_opts = {
        "outtmpl": output_path,
        "format": "best[ext=mp4]/best",
        "quiet": True,
        "noplaylist": True,
        "no_warnings": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        filename = ydl.prepare_filename(info)
        base_name = os.path.basename(filename).rsplit(".", 1)[0]
        final_path = os.path.join(DOWNLOADS_DIR, f"{base_name}.mp4")

    if not os.path.exists(final_path):
        for file in os.listdir(DOWNLOADS_DIR):
            if file.startswith(f"video_{video_id}") and file.endswith(".mp4"):
                final_path = os.path.join(DOWNLOADS_DIR, file)
                break

    if not os.path.exists(final_path):
        raise FileNotFoundError("No se pudo descargar el video.")

    return final_path

def extract_audio(video_path: str) -> str:
    wav_path = video_path.replace(".mp4", ".wav")
    cmd = [
        "ffmpeg",
        "-y",
        "-i", video_path,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        wav_path
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return wav_path

def transcribe_segments(video_path: str):
    model = whisper.load_model("base")
    audio_path = extract_audio(video_path)
    result = model.transcribe(audio_path, language="es")
    if os.path.exists(audio_path):
        os.remove(audio_path)
    return result.get("segments", [])

def create_clip(video_path: str, start: float, end: float) -> str:
    clip_id = uuid.uuid4().hex[:8]
    output_path = os.path.join(DOWNLOADS_DIR, f"clip_{clip_id}.mp4")

    start_time = max(0, start - 1.0)
    end_time = end + 1.5

    cmd = [
        "ffmpeg",
        "-y",
        "-ss", str(start_time),
        "-to", str(end_time),
        "-i", video_path,
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "28",
        "-c:a", "aac",
        "-movflags", "+faststart",
        output_path
    ]

    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return output_path

def seconds_to_duration(seconds: float) -> str:
    total = max(0, int(seconds))
    minutes, secs = divmod(total, 60)
    return f"{minutes:02d}:{secs:02d}"

def generate_clips(video_path: str):
    segments = transcribe_segments(video_path)
    if not segments:
        fallback_path = create_clip(video_path, 0, 18)
        return [{
            "title": "Momento clave",
            "duration": "00:18",
            "download_url": f"/downloads/{os.path.basename(fallback_path)}"
        }]

    candidates = []
    for seg in segments:
        text = (seg.get("text") or "").strip()
        if len(text.split()) < 4:
            continue
        start = float(seg.get("start", 0))
        end = float(seg.get("end", start + 10))
        duration = max(0.0, end - start)
        candidates.append((duration, start, end, text))

    candidates.sort(key=lambda x: x[0], reverse=True)

    selected = []
    for duration, start, end, text in candidates[:10]:
        clip_path = create_clip(video_path, start, end)
        selected.append({
            "title": text[:60].strip() or "Momento clave",
            "duration": seconds_to_duration(duration),
            "download_url": f"/downloads/{os.path.basename(clip_path)}"
        })

    if not selected:
        fallback_path = create_clip(video_path, 0, 18)
        return [{
            "title": "Momento clave",
            "duration": "00:18",
            "download_url": f"/downloads/{os.path.basename(fallback_path)}"
        }]

    return selected[:10]

@app.post("/api/start-stream")
async def start_stream(payload: StreamRequest):
    global PROCESSING, LAST_MESSAGE

    url = payload.url.strip()
    if not url:
        raise HTTPException(status_code=400, detail="La URL es requerida.")

    if not is_valid_youtube_url(url):
        raise HTTPException(status_code=400, detail="URL de YouTube inválida.")

    if PROCESSING:
        return {"message": "Ya hay un análisis en curso.", "processing": True}

    PROCESSING = True
    LAST_MESSAGE = "Procesando tu transmisión..."
    CLIPS.clear()

    try:
        video_path = download_youtube_video(url)
        clips = generate_clips(video_path)
        CLIPS.extend(clips)
        LAST_MESSAGE = "Análisis completado. Se están generando los mejores momentos."
        return {
            "message": LAST_MESSAGE,
            "processing": False
        }
    except Exception as e:
        LAST_MESSAGE = f"Error: {str(e)}"
        raise HTTPException(status_code=500, detail=LAST_MESSAGE)
    finally:
        PROCESSING = False

@app.get("/api/clips")
async def get_clips():
    return {
        "clips": CLIPS,
        "processing": PROCESSING,
        "message": LAST_MESSAGE
    }

@app.get("/")
async def root():
    return {"status": "ok", "message": "Live Clipper Pro API funcionando"}

@app.get("/health")
async def health():
    return {"status": "ok"}
    <!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    <title>Live Clipper Pro</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <meta name="apple-mobile-web-app-capable" content="yes">
    <meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
</head>
<body class="bg-black text-white min-h-screen pb-12 select-none">
    
    <header class="sticky top-0 z-50 bg-black/80 backdrop-blur-md px-4 py-4 border-b border-zinc-800 flex justify-between items-center">
        <h1 class="text-lg font-bold bg-gradient-to-r from-cyan-400 to-purple-500 bg-clip-text text-transparent">
            ⚡ Live Clipper Pro
        </h1>
        <span class="text-xs bg-zinc-800 text-zinc-400 px-2 py-1 rounded-full">Máx. 10 Clips</span>
    </header>

    <main class="p-4 max-w-md mx-auto">
        <div class="bg-zinc-900 border border-zinc-800 p-4 rounded-2xl shadow-xl mb-6">
            <label class="block text-xs font-semibold text-zinc-400 mb-2 uppercase tracking-wider">Link del En Vivo de YouTube</label>
            <input id="streamUrl" type="url" placeholder="https://youtube.com/watch?v=..." 
                   class="w-full bg-zinc-950 border border-zinc-800 rounded-xl px-4 py-3 text-sm text-white focus:outline-none focus:border-purple-500 mb-3">
            <button onclick="startAnalysis()" class="w-full bg-gradient-to-r from-purple-600 to-indigo-600 active:scale-95 hover:opacity-90 py-3 rounded-xl font-semibold text-sm transition shadow-lg shadow-purple-900/30">
                Iniciar Motor Neural 🚀
            </button>
        </div>

        <div id="statusMessage" class="text-center text-xs text-purple-400 mb-6 font-medium"></div>

        <div class="flex justify-between items-center mb-3">
            <h2 class="text-sm font-bold text-zinc-300">Clips Full HD</h2>
            <span id="clipCount" class="text-xs text-zinc-500">0 / 10</span>
        </div>
        
        <div id="clipsGrid" class="space-y-4">
            <div class="text-center py-12 text-zinc-600 text-sm">
                Esperando enlace de transmisión...
            </div>
        </div>
    </main>

    <script>
        const BACKEND_URL = "https://tu-backend.up.railway.app";
        let clipInterval = null;
        let processingStarted = false;

        function setStatus(message) {
            document.getElementById('statusMessage').innerText = message;
        }

        function buildDownloadUrl(path) {
            if (!path) return "#";
            if (/^https?:\/\//i.test(path)) return path;
            return `${BACKEND_URL}${path.startsWith("/") ? path : "/" + path}`;
        }

        function showLoadingState(message = "Procesando clips...") {
            const grid = document.getElementById('clipsGrid');
            grid.innerHTML = `
                <div class="flex items-center justify-center gap-3 py-12 text-zinc-400 text-sm">
                    <div class="w-4 h-4 border-2 border-purple-500 border-t-transparent rounded-full animate-spin"></div>
                    <span>${message}</span>
                </div>
            `;
        }

        async function startAnalysis() {
            const url = document.getElementById('streamUrl').value.trim();
            if (!url) {
                alert("Por favor pega una URL de YouTube.");
                return;
            }

            if (processingStarted) {
                return;
            }

            processingStarted = true;
            setStatus("⏳ Conectando con los núcleos de procesamiento...");
            showLoadingState("Analizando la transmisión...");

            try {
                const response = await fetch(`${BACKEND_URL}/api/start-stream`, {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ url })
                });

                const data = await response.json();

                if (!response.ok) {
                    throw new Error(data.detail || "Error del servidor");
                }

                setStatus("✨ " + (data.message || "Análisis iniciado correctamente."));
                
                if (clipInterval) clearInterval(clipInterval);
                clipInterval = setInterval(fetchClips, 3000);
                await fetchClips();
            } catch (error) {
                console.error(error);
                setStatus("❌ Error de conexión con el servidor.");
                processingStarted = false;
            }
        }

        async function fetchClips() {
            try {
                const response = await fetch(`${BACKEND_URL}/api/clips`);
                if (!response.ok) throw new Error("No se pudieron cargar los clips");

                const data = await response.json();
                const clips = Array.isArray(data.clips) ? data.clips : [];
                const processing = !!data.processing;
                const statusText = data.message || "";

                const countSpan = document.getElementById('clipCount');
                const grid = document.getElementById('clipsGrid');

                countSpan.innerText = `${clips.length} / 10`;

                if (processing) {
                    setStatus("⏳ " + (statusText || "Procesando clips..."));
                    showLoadingState(statusText || "Procesando clips...");
                    return;
                }

                if (clips.length === 0) {
                    setStatus("Esperando enlaces...");
                    grid.innerHTML = `
                        <div class="text-center py-12 text-zinc-600 text-sm">
                            Esperando enlace de transmisión...
                        </div>
                    `;
                    return;
                }

                setStatus("✨ " + (statusText || "Clips listos."));
                grid.innerHTML = "";

                clips.forEach(clip => {
                    const card = document.createElement("div");
                    card.className = "bg-zinc-900 border border-zinc-800 p-4 rounded-2xl flex items-center justify-between shadow-md";
                    card.innerHTML = `
                        <div>
                            <h3 class="font-semibold text-sm text-white">${clip.title || "Clip generado"}</h3>
                            <p class="text-xs text-zinc-500 mt-0.5">Vertical 9:16 • ${clip.duration || "00:00"}</p>
                        </div>
                        <a href="${buildDownloadUrl(clip.download_url)}" target="_blank" class="bg-zinc-800 hover:bg-purple-600 text-purple-400 hover:text-white p-3 rounded-xl transition text-xs font-semibold flex items-center gap-1">
                            📥 Descargar
                        </a>
                    `;
                    grid.appendChild(card);
                });

                processingStarted = false;
                if (clipInterval) clearInterval(clipInterval);
                clipInterval = null;

            } catch (error) {
                console.log("Actualizando clips...");
            }
        }
    </script>
</body>
</html>
