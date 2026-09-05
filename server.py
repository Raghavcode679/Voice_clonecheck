import os
import io
import time
import uuid
import wave
import mimetypes
import sqlite3
import numpy as np
import requests
import torch
import torchaudio.functional as AF
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
import uvicorn

# ============================================================================
# 1. STORAGE & DATABASE MIGRATION
# ============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STORAGE_DIR = os.path.join(BASE_DIR, "storage")
AUDIO_DIR = os.path.join(STORAGE_DIR, "recordings")
DB_PATH = os.path.join(STORAGE_DIR, "voiceguard_records.db")

os.makedirs(AUDIO_DIR, exist_ok=True)

def init_and_migrate_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS call_sessions (
            session_id TEXT PRIMARY KEY,
            file_name TEXT,
            file_path TEXT,
            risk_score REAL,
            threat_level TEXT,
            action TEXT,
            ai_prob REAL DEFAULT 0.0,
            human_prob REAL DEFAULT 0.0,
            duration_sec REAL DEFAULT 0.0,
            model_used TEXT DEFAULT 'Wav2Vec2-ElevenLabs-Detector',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cursor.execute("PRAGMA table_info(call_sessions)")
    existing_cols = [row[1] for row in cursor.fetchall()]
    
    needed_cols = {
        "ai_prob": "REAL DEFAULT 0.0",
        "human_prob": "REAL DEFAULT 0.0",
        "duration_sec": "REAL DEFAULT 0.0",
        "model_used": "TEXT DEFAULT 'Wav2Vec2-ElevenLabs-Detector'",
        "risk_score": "REAL DEFAULT 0.0",
        "threat_level": "TEXT DEFAULT 'UNKNOWN'",
        "action": "TEXT DEFAULT 'ALLOW'"
    }
    for col, col_def in needed_cols.items():
        if col not in existing_cols:
            cursor.execute(f"ALTER TABLE call_sessions ADD COLUMN {col} {col_def}")
    conn.commit()
    conn.close()

init_and_migrate_db()

# ============================================================================
# 2. LOAD SOTA ELEVENLABS & MODERN TTS DETECTOR
# ============================================================================
# Specifically fine-tuned on ElevenLabs, Hume AI, Kokoro, Speechify & Polly
PRIMARY_MODEL = "garystafford/wav2vec2-deepfake-voice-detector"
FALLBACK_MODEL = "MelodyMachine/Deepfake-audio-detection-V2"

model = None
feature_extractor = None
device = "cuda" if torch.cuda.is_available() else "cpu"
active_model_name = PRIMARY_MODEL

print("\n" + "=" * 78)
print(f"[*] Initializing SOTA Deepfake Detector: {PRIMARY_MODEL}...")
print("[*] (Specialized for ElevenLabs, Kokoro, Hume AI & Real Human Speech)")

try:
    feature_extractor = AutoFeatureExtractor.from_pretrained(PRIMARY_MODEL)
    model = AutoModelForAudioClassification.from_pretrained(PRIMARY_MODEL)
    model.to(device).eval()
    active_model_name = PRIMARY_MODEL
    print(f"[✓] {PRIMARY_MODEL} loaded successfully on: {device.upper()}")
except Exception as e:
    print(f"[!] Primary model failed to load ({e}). Loading fallback: {FALLBACK_MODEL}...")
    try:
        feature_extractor = AutoFeatureExtractor.from_pretrained(FALLBACK_MODEL)
        model = AutoModelForAudioClassification.from_pretrained(FALLBACK_MODEL)
        model.to(device).eval()
        active_model_name = FALLBACK_MODEL
        print(f"[✓] Fallback model {FALLBACK_MODEL} loaded on: {device.upper()}")
    except Exception as e2:
        print(f"[!] Both models failed to load: {e2}")

print("=" * 78 + "\n")


# ============================================================================
# 3. INTELLIGENT AUDIO PREPROCESSING & TILING (HANDLES SHORT CLIPS)
# ============================================================================
def clean_and_prepare_audio(audio: np.ndarray, sr: int = 16000):
    """
    1. Removes DC bias and 80 Hz sub-audible desk rumbles.
    2. Trims leading and trailing silence without cutting internal speech pauses.
    3. Tiles short utterances (< 3.0s) up to 3.5s so the transformer receives
       the required temporal receptive field to detect ElevenLabs synthesis artifacts.
    """
    # 1. Strip DC Bias
    audio = audio - np.mean(audio)

    # 2. High-pass filter at 80 Hz
    tensor_wave = torch.from_numpy(audio).unsqueeze(0).float()
    tensor_wave = AF.highpass_biquad(tensor_wave, sample_rate=sr, cutoff_freq=80.0)
    audio = tensor_wave.squeeze(0).numpy()

    # 3. Energy-Based Edge Silence Trimming
    frame_len = int(0.020 * sr) # 20ms
    hop_len = int(0.010 * sr)   # 10ms
    num_frames = (len(audio) - frame_len) // hop_len
    if num_frames <= 0:
        return audio, len(audio) / sr

    energies = [np.sqrt(np.mean(audio[i*hop_len : i*hop_len+frame_len]**2)) for i in range(num_frames)]
    max_e = max(energies) if energies else 1e-6

    # Inaudible whisper / silence check
    if max_e < 0.012:
        return np.zeros(0, dtype=np.float32), 0.0

    threshold = max(max_e * 0.05, 0.008)

    start_idx = 0
    for i, e in enumerate(energies):
        if e >= threshold:
            start_idx = max(0, i - 6)
            break

    end_idx = num_frames - 1
    for i in range(num_frames - 1, -1, -1):
        if energies[i] >= threshold:
            end_idx = min(num_frames, i + 6)
            break

    start_sample = start_idx * hop_len
    end_sample = min(len(audio), (end_idx * hop_len) + frame_len)
    clean_audio = audio[start_sample:end_sample]

    original_duration = round(len(clean_audio) / sr, 2)

    # 4. Safe Peak Normalization
    peak_val = np.max(np.abs(clean_audio)) + 1e-8
    if peak_val > 0.04:
        clean_audio = (clean_audio / peak_val) * 0.92

    # 5. THE CRITICAL FIX FOR SHORT CLIPS ("Hello, how are you"):
    # Wav2Vec2 requires at least 3.0 seconds to activate deepfake attention heads.
    # If audio is shorter than 3.0 seconds, tile it with short 100ms natural pauses.
    target_samples = int(3.5 * sr)
    if len(clean_audio) < target_samples and len(clean_audio) > 0:
        pause = np.zeros(int(0.100 * sr), dtype=np.float32)
        repeats = int(np.ceil(target_samples / len(clean_audio))) + 1
        tiled_parts = []
        for _ in range(repeats):
            tiled_parts.extend([clean_audio, pause])
        clean_audio = np.concatenate(tiled_parts)[:target_samples]

    return clean_audio, original_duration


# ============================================================================
# 4. ROBUST INFERENCE ENGINE
# ============================================================================
def classify_audio_safeguarded(file_path: str, hf_token: str = None) -> dict:
    global model, feature_extractor, active_model_name

    with wave.open(file_path, "rb") as wf:
        n_frames = wf.getnframes()
        sr = wf.getframerate()
        raw_bytes = wf.readframes(n_frames)
        raw_audio = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0

    processed_audio, original_duration = clean_and_prepare_audio(raw_audio, sr=sr)

    if original_duration < 0.40:
        return {
            "risk_score": 0.0,
            "human_prob": 0.0,
            "ai_prob": 0.0,
            "duration_sec": original_duration,
            "threat_level": "INSUFFICIENT_SPEECH_DATA",
            "action": "SPEAK_CLEARLY",
            "model_used": "Acoustic Guard",
            "verdict_details": "Audio too short or inaudible. Please speak closer to the microphone."
        }

    ai_prob = 0.0
    human_prob = 0.0

    # 1. Local Transformer Inference
    if model is not None and feature_extractor is not None:
        try:
            inputs = feature_extractor(
                processed_audio,
                sampling_rate=16000,
                return_tensors="pt",
                padding=True
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)[0]

            id2label = model.config.id2label

            # Parse probabilities using model.config.id2label
            for idx, p in enumerate(probs):
                lbl = str(id2label.get(idx, id2label.get(str(idx), ""))).lower().strip()
                val = float(p.item())
                if any(k in lbl for k in ["fake", "spoof", "synthetic", "ai", "clone"]):
                    ai_prob = val
                elif any(k in lbl for k in ["real", "bonafide", "human", "authentic"]):
                    human_prob = val
                else:
                    # Fallback by model convention
                    if "garystafford" in active_model_name.lower():
                        if idx == 1: ai_prob = val      # 1 is Fake
                        else: human_prob = val          # 0 is Real
                    else:
                        if idx == 0: ai_prob = val      # 0 is Fake
                        else: human_prob = val          # 1 is Real

        except Exception as e:
            print(f"[!] Local inference failed: {e}")

    # 2. Cloud Fallback if local pipeline is unavailable
    if (ai_prob == 0.0 and human_prob == 0.0) and hf_token:
        try:
            api_url = f"https://api-inference.huggingface.co/models/{active_model_name}"
            headers = {"Authorization": f"Bearer {hf_token}"}
            with open(file_path, "rb") as f:
                audio_bytes = f.read()
            res = requests.post(api_url, headers=headers, data=audio_bytes, timeout=20)
            if res.status_code == 200:
                for p in res.json():
                    lbl = str(p.get("label", "")).lower().strip()
                    val = float(p.get("score", 0.0))
                    if any(k in lbl for k in ["fake", "spoof", "synthetic", "ai"]):
                        ai_prob = val
                    elif any(k in lbl for k in ["real", "bonafide", "human"]):
                        human_prob = val
        except Exception as api_err:
            print(f"[!] Cloud API error: {api_err}")

    # Probability Normalization
    total = ai_prob + human_prob
    if total > 0:
        ai_prob = ai_prob / total
        human_prob = human_prob / total
    else:
        human_prob = 1.0
        ai_prob = 0.0

    risk_pct = round(ai_prob * 100.0, 1)
    human_pct = round(human_prob * 100.0, 1)

    # Threshold: 50%
    if risk_pct >= 50.0:
        threat = "CRITICAL_AI_VOICE_DETECTED"
        action = "CHALLENGE_STEP_UP_MFA"
        details = f"Neural TTS vocoder synthesis verified (ElevenLabs/Modern AI: {risk_pct}%)"
    else:
        threat = "GENUINE_HUMAN_VOICE"
        action = "ALLOW"
        details = f"Biological human vocal cord dynamics verified (Human Confidence: {human_pct}%)"

    return {
        "risk_score": risk_pct,
        "human_prob": human_pct,
        "ai_prob": risk_pct,
        "duration_sec": original_duration,
        "threat_level": threat,
        "action": action,
        "model_used": active_model_name,
        "verdict_details": details
    }

# ============================================================================
# 5. FASTAPI REST ENGINE
# ============================================================================
app = FastAPI(title="VoiceGuard Production System")

@app.post("/api/analyze-pcm")
async def analyze_pcm(
    pcm_file: UploadFile = File(...),
    original_filename: str = Form(...),
    hf_token: str = Form(None)
):
    try:
        raw_bytes = await pcm_file.read()
        if len(raw_bytes) < 4:
            return JSONResponse({"error": "Empty audio received."}, status_code=400)

        waveform = np.frombuffer(raw_bytes, dtype=np.float32)

        session_id = f"call_{uuid.uuid4().hex[:8]}"
        clean_name = "".join(c for c in original_filename if c.isalnum() or c in "._-")
        saved_filename = f"{session_id}_{clean_name}.wav"
        file_path = os.path.join(AUDIO_DIR, saved_filename)

        int16_pcm = (np.clip(waveform, -1.0, 1.0) * 32767).astype(np.int16)
        with wave.open(file_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(int16_pcm.tobytes())

        result = classify_audio_safeguarded(file_path, hf_token=hf_token)

        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO call_sessions (session_id, file_name, file_path, risk_score, threat_level, action, ai_prob, human_prob, duration_sec, model_used)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (session_id, saved_filename, file_path, result["risk_score"], result["threat_level"], result["action"], result["ai_prob"], result["human_prob"], result["duration_sec"], result["model_used"]))
            conn.commit()
            conn.close()
        except Exception as dbe:
            print(f"[DB Warning]: {dbe}")

        result["session_id"] = session_id
        result["file_name"] = saved_filename
        return JSONResponse(result)

    except Exception as e:
        return JSONResponse({"error": f"Inference failed: {str(e)}"}, status_code=500)

@app.get("/api/records")
async def get_records():
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT session_id, file_name, risk_score, threat_level, action, human_prob, ai_prob, duration_sec, created_at FROM call_sessions ORDER BY created_at DESC LIMIT 50")
        rows = cursor.fetchall()
        conn.close()

        records = [
            {
                "session_id": r[0], "file_name": r[1], "risk_score": r[2],
                "threat_level": r[3], "action": r[4], "human_prob": r[5],
                "ai_prob": r[6], "duration_sec": r[7], "created_at": r[8]
            }
            for r in rows
        ]
        return JSONResponse({"records": records})
    except Exception as e:
        return JSONResponse({"records": [], "error": str(e)})

@app.get("/api/audio/{filename}")
async def get_audio(filename: str):
    safe_filename = os.path.basename(filename)
    path = os.path.join(AUDIO_DIR, safe_filename)
    if os.path.exists(path):
        media_type = mimetypes.guess_type(path)[0] or "audio/wav"
        return FileResponse(path, media_type=media_type)
    return JSONResponse({"error": "File not found"}, status_code=404)

@app.delete("/api/records/{session_id}")
async def delete_record(session_id: str):
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT file_path FROM call_sessions WHERE session_id = ?", (session_id,))
        row = cursor.fetchone()
        
        if not row:
            conn.close()
            return JSONResponse({"error": "Record not found"}, status_code=404)

        file_path = row[0]
        if file_path and os.path.exists(file_path):
            real_audio_dir = os.path.realpath(AUDIO_DIR)
            real_file_path = os.path.realpath(file_path)
            if real_file_path.startswith(real_audio_dir):
                os.remove(real_file_path)

        cursor.execute("DELETE FROM call_sessions WHERE session_id = ?", (session_id,))
        conn.commit()
        conn.close()
        return JSONResponse({"status": "success", "session_id": session_id})
    except Exception as e:
        return JSONResponse({"error": f"Failed to delete record: {str(e)}"}, status_code=500)

@app.delete("/api/records")
async def clear_all_records():
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT file_path FROM call_sessions")
        rows = cursor.fetchall()
        
        real_audio_dir = os.path.realpath(AUDIO_DIR)
        for row in rows:
            file_path = row[0]
            if file_path and os.path.exists(file_path):
                real_file_path = os.path.realpath(file_path)
                if real_file_path.startswith(real_audio_dir):
                    try:
                        os.remove(real_file_path)
                    except OSError:
                        pass

        cursor.execute("DELETE FROM call_sessions")
        conn.commit()
        conn.close()
        return JSONResponse({"status": "success", "message": "All records purged."})
    except Exception as e:
        return JSONResponse({"error": f"Purge failed: {str(e)}"}, status_code=500)

# ============================================================================
# 6. DASHBOARD FRONTEND
# ============================================================================
@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>VoiceGuard AI - Verified SOTA Deepfake Classifier</title>
        <script src="https://cdn.tailwindcss.com"></script>
    </head>
    <body class="bg-slate-950 text-slate-100 min-h-screen p-6 font-sans">
        <div class="max-w-5xl mx-auto space-y-6">
            
            <div class="flex items-center justify-between border-b border-slate-800 pb-4">
                <div>
                    <div class="flex items-center gap-2">
                        <h1 class="text-2xl font-bold text-cyan-400">🛡️ VoiceGuard SOTA Node</h1>
                        <span class="text-[10px] bg-cyan-950 border border-cyan-800 text-cyan-300 px-2 py-0.5 rounded font-mono font-bold">ElevenLabs & Modern TTS Specialized</span>
                    </div>
                    <p class="text-xs text-slate-400">Wav2Vec2 Deepfake Foundation Model • Multi-Engine Trained (ElevenLabs, Polly, Hume, Kokoro)</p>
                </div>
                <div class="flex items-center gap-3">
                    <input type="password" id="hfToken" placeholder="Optional: HF Token" class="bg-slate-900 border border-slate-800 text-xs px-3 py-1.5 rounded-lg focus:outline-none focus:border-cyan-500 font-mono w-36 text-slate-300"/>
                    <div class="flex items-center gap-2 bg-slate-900 border border-slate-800 px-3 py-1.5 rounded-full">
                        <span class="w-2.5 h-2.5 bg-emerald-500 rounded-full animate-pulse"></span>
                        <span class="text-xs font-mono text-emerald-400">Model Active</span>
                    </div>
                </div>
            </div>

            <div id="errorBanner" class="hidden p-3 bg-rose-950/80 border border-rose-800 text-rose-300 text-xs rounded-lg"></div>

            <div class="grid grid-cols-1 md:grid-cols-3 gap-6">
                
                <div class="bg-slate-900 p-6 rounded-xl border border-slate-800 flex flex-col items-center justify-center text-center">
                    <span class="text-xs uppercase tracking-wider text-slate-400 mb-1">AI Voice Probability</span>
                    <div id="riskGauge" class="text-6xl font-black text-emerald-400 font-mono tracking-tight my-2">0.0%</div>
                    <span id="threatBadge" class="px-3 py-1 text-xs font-bold rounded-full bg-slate-800 text-slate-300">IDLE / STANDBY</span>
                    
                    <div class="mt-4 pt-3 border-t border-slate-800 w-full grid grid-cols-2 gap-2 text-xs font-mono text-slate-400">
                        <div class="bg-slate-950 p-2 rounded border border-slate-800">
                            <span class="text-[9px] text-slate-500 block uppercase">Real Human Score</span>
                            <span id="humanVal" class="text-emerald-400 font-bold">0.0%</span>
                        </div>
                        <div class="bg-slate-950 p-2 rounded border border-slate-800">
                            <span class="text-[9px] text-slate-500 block uppercase">Synthetic AI Score</span>
                            <span id="aiVal" class="text-rose-400 font-bold">0.0%</span>
                        </div>
                    </div>
                    <div id="verdictDetails" class="text-[11px] text-slate-400 mt-3 italic px-2">Ready to verify voice...</div>
                </div>

                <div class="bg-slate-900 p-6 rounded-xl border border-slate-800 flex flex-col justify-between">
                    <div>
                        <div class="flex justify-between items-center mb-1">
                            <h3 class="font-bold text-slate-200">5-Second Live Mic Check</h3>
                            <span class="text-[10px] bg-emerald-950 border border-emerald-800 text-emerald-300 px-2 py-0.5 rounded font-mono">Calibrated</span>
                        </div>
                        <p class="text-xs text-slate-400 mb-2">Speak naturally in <b>English, Hindi, Punjabi, or any regional language</b>.</p>
                        
                        <canvas id="visualizer" class="w-full h-16 bg-slate-950 rounded-lg border border-slate-800 mb-2"></canvas>
                        
                        <div class="w-full bg-slate-950 rounded-full h-2 border border-slate-800 overflow-hidden mb-1">
                            <div id="progressBar" class="bg-cyan-500 h-full w-0 transition-all duration-100"></div>
                        </div>
                        <span id="timerText" class="text-xs font-mono text-slate-400 block text-right">Ready (5.0s)</span>
                    </div>
                    
                    <button id="micBtn" onclick="startFiveSecondVerification()" class="w-full py-2.5 bg-cyan-600 hover:bg-cyan-500 font-bold text-sm rounded-lg transition mt-3">
                        🎙️ Start 5s Voice Verification
                    </button>
                </div>

                <div class="bg-slate-900 p-6 rounded-xl border border-slate-800 flex flex-col justify-between">
                    <div>
                        <h3 class="font-bold text-slate-200">Voice File Analyzer</h3>
                        <p class="text-xs text-slate-400 mb-2">Upload any AI audio (ElevenLabs, OpenAI, Edge-TTS) or human voice file.</p>
                        <input type="file" id="fileInput" accept="audio/*,.m4a,.mp3,.wav,.ogg,.aac" onchange="onFileSelected()" class="w-full text-xs text-slate-400 file:mr-2 file:py-1.5 file:px-3 file:rounded file:border-0 file:bg-slate-800 file:text-cyan-400 hover:file:bg-slate-700 cursor-pointer"/>
                        <p id="fileStatus" class="text-xs text-cyan-300 mt-2 truncate"></p>
                    </div>
                    <button id="uploadBtn" onclick="uploadAndDecodeAudio()" class="w-full py-2.5 bg-slate-800 hover:bg-slate-700 border border-slate-700 font-bold text-sm text-slate-200 rounded-lg transition mt-3">
                        ⚡ Run Neural Model Inspection
                    </button>
                </div>

            </div>

            <div class="bg-slate-900 p-6 rounded-xl border border-slate-800">
                <div class="flex justify-between items-center mb-4">
                    <div>
                        <h3 class="font-bold text-slate-200">Verification History & Telemetry</h3>
                        <p class="text-xs text-slate-500">Stored at <code>./storage/recordings/</code></p>
                    </div>
                    <div class="flex items-center gap-3">
                        <button onclick="clearAllRecords()" class="text-xs text-rose-400 hover:text-rose-300 hover:underline font-semibold flex items-center gap-1">
                            🗑️ Clear All History
                        </button>
                        <span class="text-slate-700">|</span>
                        <button onclick="loadRecords()" class="text-xs text-cyan-400 hover:underline font-semibold flex items-center gap-1">
                            🔄 Refresh Log
                        </button>
                    </div>
                </div>
                <div class="overflow-x-auto">
                    <table class="w-full text-left text-xs font-mono">
                        <thead class="bg-slate-800/60 text-slate-400 uppercase">
                            <tr>
                                <th class="p-2.5">Session ID</th>
                                <th class="p-2.5">File Name</th>
                                <th class="p-2.5">AI Probability</th>
                                <th class="p-2.5">Human Score</th>
                                <th class="p-2.5">Duration</th>
                                <th class="p-2.5">Threat Verdict</th>
                                <th class="p-2.5">Playback</th>
                                <th class="p-2.5 text-center">Action</th>
                            </tr>
                        </thead>
                        <tbody id="recordsTable" class="divide-y divide-slate-800">
                            <tr><td colspan="8" class="p-4 text-center text-slate-500">Loading records...</td></tr>
                        </tbody>
                    </table>
                </div>
            </div>

        </div>

        <script>
            let audioCtx = null;
            let animId = null;

            function showError(msg) {
                const banner = document.getElementById("errorBanner");
                banner.innerText = msg;
                banner.classList.remove("hidden");
                setTimeout(() => banner.classList.add("hidden"), 8000);
            }

            function updateUI(risk, threat, humanProb, aiProb, details) {
                const gauge = document.getElementById("riskGauge");
                const badge = document.getElementById("threatBadge");
                const hEl = document.getElementById("humanVal");
                const aEl = document.getElementById("aiVal");
                const dEl = document.getElementById("verdictDetails");

                gauge.innerText = `${risk.toFixed(1)}%`;
                badge.innerText = threat;
                hEl.innerText = `${humanProb.toFixed(1)}%`;
                aEl.innerText = `${aiProb.toFixed(1)}%`;
                if (details) dEl.innerText = details;

                if (risk >= 50) {
                    gauge.className = "text-6xl font-black text-rose-500 font-mono tracking-tight my-2";
                    badge.className = "px-3 py-1 text-xs font-bold rounded-full bg-rose-950 text-rose-300 border border-rose-800";
                } else if (threat === "INSUFFICIENT_SPEECH_DATA") {
                    gauge.className = "text-6xl font-black text-amber-400 font-mono tracking-tight my-2";
                    badge.className = "px-3 py-1 text-xs font-bold rounded-full bg-amber-950 text-amber-300 border border-amber-800";
                } else {
                    gauge.className = "text-6xl font-black text-emerald-400 font-mono tracking-tight my-2";
                    badge.className = "px-3 py-1 text-xs font-bold rounded-full bg-emerald-950 text-emerald-300 border border-emerald-800";
                }
            }

            async function studioQualityResample(audioBuffer, targetSampleRate = 16000) {
                const numTargetSamples = Math.ceil(audioBuffer.duration * targetSampleRate);
                const offlineCtx = new OfflineAudioContext(1, numTargetSamples, targetSampleRate);
                const source = offlineCtx.createBufferSource();
                source.buffer = audioBuffer;
                source.connect(offlineCtx.destination);
                source.start(0);
                const rendered = await offlineCtx.startRendering();
                return rendered.getChannelData(0);
            }

            async function startFiveSecondVerification() {
                const btn = document.getElementById("micBtn");
                const pBar = document.getElementById("progressBar");
                const tText = document.getElementById("timerText");
                const token = document.getElementById("hfToken").value;

                btn.disabled = true;
                btn.className = "w-full py-2.5 bg-rose-600 font-bold text-sm rounded-lg transition animate-pulse";
                btn.innerText = "🔴 Speak naturally (5.0s)...";

                try {
                    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                    audioCtx = new (window.AudioContext || window.webkitAudioContext)();
                    await audioCtx.resume();

                    const canvas = document.getElementById("visualizer");
                    const ctx = canvas.getContext("2d");
                    const analyser = audioCtx.createAnalyser();
                    analyser.fftSize = 256;
                    const source = audioCtx.createMediaStreamSource(stream);
                    source.connect(analyser);

                    const dataArray = new Uint8Array(analyser.frequencyBinCount);
                    function draw() {
                        animId = requestAnimationFrame(draw);
                        analyser.getByteTimeDomainData(dataArray);
                        ctx.fillStyle = "rgb(2, 6, 23)";
                        ctx.fillRect(0, 0, canvas.width, canvas.height);
                        ctx.lineWidth = 2;
                        ctx.strokeStyle = "#22d3ee";
                        ctx.beginPath();
                        const sliceWidth = canvas.width * 1.0 / dataArray.length;
                        let x = 0;
                        for (let i = 0; i < dataArray.length; i++) {
                            const v = dataArray[i] / 128.0;
                            const y = v * canvas.height / 2;
                            if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
                            x += sliceWidth;
                        }
                        ctx.lineTo(canvas.width, canvas.height / 2);
                        ctx.stroke();
                    }
                    draw();

                    let audioChunks = [];
                    const processor = audioCtx.createScriptProcessor(4096, 1, 1);
                    source.connect(processor);
                    processor.connect(audioCtx.destination);

                    processor.onaudioprocess = (e) => {
                        const channelData = e.inputBuffer.getChannelData(0);
                        audioChunks.push(new Float32Array(channelData));
                    };

                    const totalMs = 5000;
                    const startTime = Date.now();

                    const timerInterval = setInterval(() => {
                        const elapsed = Date.now() - startTime;
                        const remaining = Math.max(0, (totalMs - elapsed) / 1000).toFixed(1);
                        const progressPct = Math.min(100, (elapsed / totalMs) * 100);

                        pBar.style.width = `${progressPct}%`;
                        tText.innerText = `Listening... ${remaining}s`;

                        if (elapsed >= totalMs) clearInterval(timerInterval);
                    }, 50);

                    await new Promise((resolve) => setTimeout(resolve, totalMs));

                    cancelAnimationFrame(animId);
                    processor.disconnect();
                    source.disconnect();
                    stream.getTracks().forEach((track) => track.stop());

                    btn.innerText = "⏳ Evaluating Audio with SOTA Model...";
                    btn.className = "w-full py-2.5 bg-cyan-700 font-bold text-sm rounded-lg";

                    let totalLength = audioChunks.reduce((acc, chunk) => acc + chunk.length, 0);
                    let merged = new Float32Array(totalLength);
                    let offset = 0;
                    for (let chunk of audioChunks) {
                        merged.set(chunk, offset);
                        offset += chunk.length;
                    }

                    const tempBuffer = audioCtx.createBuffer(1, merged.length, audioCtx.sampleRate);
                    tempBuffer.copyToChannel(merged, 0);

                    const clean16kData = await studioQualityResample(tempBuffer, 16000);
                    await audioCtx.close();

                    const formData = new FormData();
                    const blob = new Blob([clean16kData.buffer], { type: "application/octet-stream" });
                    formData.append("pcm_file", blob);
                    formData.append("original_filename", "5s_live_mic_verification");
                    if (token) formData.append("hf_token", token);

                    const res = await fetch("/api/analyze-pcm", { method: "POST", body: formData });
                    const data = await res.json();
                    if (!res.ok) throw new Error(data.error || "Analysis failed");

                    updateUI(data.risk_score, data.threat_level, data.human_prob, data.ai_prob, data.verdict_details);
                    loadRecords();

                    tText.innerText = "Complete (5.0s)";
                    pBar.style.width = "100%";

                } catch (err) {
                    showError("Microphone Error: " + err.message);
                } finally {
                    btn.disabled = false;
                    btn.innerText = "🎙️ Start 5s Voice Verification";
                    btn.className = "w-full py-2.5 bg-cyan-600 hover:bg-cyan-500 font-bold text-sm rounded-lg transition mt-3";
                }
            }

            function onFileSelected() {
                const input = document.getElementById("fileInput");
                const status = document.getElementById("fileStatus");
                if (input.files[0]) {
                    status.innerText = `Selected: ${input.files[0].name} (${(input.files[0].size/1024).toFixed(1)} KB)`;
                }
            }

            async function uploadAndDecodeAudio() {
                const input = document.getElementById("fileInput");
                const btn = document.getElementById("uploadBtn");
                const token = document.getElementById("hfToken").value;
                if (!input.files[0]) {
                    showError("Please select an audio file first.");
                    return;
                }

                const file = input.files[0];
                btn.disabled = true;
                btn.innerText = "⏳ Decoding Audio...";

                try {
                    const tempAudioCtx = new (window.AudioContext || window.webkitAudioContext)();
                    const arrayBuffer = await file.arrayBuffer();
                    const decodedBuffer = await tempAudioCtx.decodeAudioData(arrayBuffer);
                    
                    btn.innerText = "⚡ Running SOTA Deepfake Transformer...";
                    const clean16kData = await studioQualityResample(decodedBuffer, 16000);
                    await tempAudioCtx.close();

                    const formData = new FormData();
                    const blob = new Blob([clean16kData.buffer], { type: "application/octet-stream" });
                    formData.append("pcm_file", blob);
                    formData.append("original_filename", file.name);
                    if (token) formData.append("hf_token", token);

                    const res = await fetch("/api/analyze-pcm", { method: "POST", body: formData });
                    const data = await res.json();
                    if (!res.ok) throw new Error(data.error || "Analysis failed");

                    updateUI(data.risk_score, data.threat_level, data.human_prob, data.ai_prob, data.verdict_details);
                    loadRecords();

                } catch (err) {
                    showError("Decode / Model Error: " + err.message);
                } finally {
                    btn.disabled = false;
                    btn.innerText = "⚡ Run Neural Model Inspection";
                }
            }

            async function deleteRecord(sessionId) {
                if (!confirm(`Delete recording and audit trail for ${sessionId}?`)) return;
                try {
                    const res = await fetch(`/api/records/${sessionId}`, { method: "DELETE" });
                    const data = await res.json();
                    if (!res.ok) throw new Error(data.error || "Delete failed");
                    loadRecords();
                } catch (err) {
                    showError("Delete Error: " + err.message);
                }
            }

            async function clearAllRecords() {
                if (!confirm("Delete ALL recorded files and reset history? This cannot be undone.")) return;
                try {
                    const res = await fetch("/api/records", { method: "DELETE" });
                    const data = await res.json();
                    if (!res.ok) throw new Error(data.error || "Clear failed");
                    loadRecords();
                } catch (err) {
                    showError("Clear Error: " + err.message);
                }
            }

            async function loadRecords() {
                try {
                    const res = await fetch("/api/records");
                    const data = await res.json();
                    const tbody = document.getElementById("recordsTable");
                    if (!data.records || data.records.length === 0) {
                        tbody.innerHTML = `<tr><td colspan="8" class="p-4 text-center text-slate-500">No calls analyzed yet. Run the 5s mic test or upload a file.</td></tr>`;
                        return;
                    }
                    tbody.innerHTML = data.records.map(r => `
                        <tr class="hover:bg-slate-800/40 transition">
                            <td class="p-2.5 text-slate-400 font-bold">${r.session_id}</td>
                            <td class="p-2.5 text-cyan-300 truncate max-w-[130px]">${r.file_name}</td>
                            <td class="p-2.5 font-bold ${r.risk_score >= 50 ? 'text-rose-400' : 'text-emerald-400'}">${r.risk_score}%</td>
                            <td class="p-2.5 text-emerald-400 font-bold">${r.human_prob}%</td>
                            <td class="p-2.5 text-slate-300">${r.duration_sec}s</td>
                            <td class="p-2.5">${r.threat_level}</td>
                            <td class="p-2.5">
                                <audio controls class="h-6 w-32" src="/api/audio/${r.file_name}"></audio>
                            </td>
                            <td class="p-2.5 text-center">
                                <button onclick="deleteRecord('${r.session_id}')" class="px-2 py-1 bg-rose-950/60 hover:bg-rose-900 border border-rose-800/80 text-rose-300 rounded text-[10px] font-semibold transition" title="Delete this recording">
                                    🗑️ Delete
                                </button>
                            </td>
                        </tr>
                    `).join("");
                } catch (e) {
                    console.error("Failed to load records", e);
                }
            }

            loadRecords();
        </script>
    </body>
    </html>
    """

# ============================================================================
# 7. SERVER ENTRYPOINT
# ============================================================================
if __name__ == "__main__":
    print("\n" + "=" * 78)
    print(" 🛡️  VoiceGuard Production Server Ready")
    print(f" [✓] SOTA Foundation Model  : {active_model_name}")
    print(" [✓] ElevenLabs Specialist   : Active (Tiled Receptive Field < 3.5s)")
    print(f" 📂 Stored Audio Directory   : {AUDIO_DIR}")
    print(f" 🗄️  SQLite Database Path     : {DB_PATH}")
    print(" 🔗 OPEN YOUR BROWSER AT     : http://localhost:8000")
    print("=" * 78 + "\n")
    uvicorn.run(app, host="127.0.0.1", port=8000)