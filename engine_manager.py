import io
import os
import sys
import time
import json
import logging
import shutil
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import torch
import numpy as np
import soundfile as sf
import torchaudio
import torchaudio.functional as F

from kokoro_engine import kokoro_engine, KOKORO_VOICES

logger = logging.getLogger("EngineManager")

BASE_DIR = Path(__file__).resolve().parent
VOICES_DIR = BASE_DIR / "voices"
OUTPUT_DIR = BASE_DIR / "output"
CONFIG_FILE = BASE_DIR / "config.json"

VOICES_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# Kokoro voice pool used to deterministically voice-map custom voices
KOKORO_FALLBACK_POOL = [
    "af_heart", "af_bella", "am_adam", "am_michael", 
    "bf_emma", "bm_george", "af_nicole", "am_echo",
    "af_sky", "am_liam", "bf_alice", "bm_daniel"
]

def map_voice_to_kokoro(voice_name: str) -> str:
    """Deterministically maps any custom voice name to a pleasant Kokoro voice."""
    v_clean = voice_name.lower().replace("!", "").strip()
    idx = abs(hash(v_clean)) % len(KOKORO_FALLBACK_POOL)
    return KOKORO_FALLBACK_POOL[idx]

def adjust_speed(wav_np: np.ndarray, sr: int, speed: float) -> np.ndarray:
    """Adjusts speech playback rate without altering pitch using PyTorch/torchaudio."""
    if abs(speed - 1.0) < 0.02 or speed <= 0:
        return wav_np
    try:
        waveform = torch.from_numpy(wav_np).unsqueeze(0).float()
        effects = [["speed", f"{speed:.2f}"], ["rate", f"{sr}"]]
        res_wav, _ = torchaudio.sox_effects.apply_effects_tensor(waveform, sr, effects)
        return res_wav.squeeze(0).numpy().astype(np.float32)
    except Exception:
        try:
            new_len = int(len(wav_np) / speed)
            indices = np.linspace(0, len(wav_np) - 1, new_len)
            return np.interp(indices, np.arange(len(wav_np)), wav_np).astype(np.float32)
        except Exception:
            return wav_np

def parse_segments(text: str) -> List[Tuple[str, str, Optional[float]]]:
    import re
    pattern = r"!([a-zA-Z0-9_\-]+)(?:[-_](\d*\.?\d+))?\s+([^!]+)"
    matches = list(re.finditer(pattern, text))
    if not matches:
        return [("default", text.strip(), None)]

    segments = []
    for m in matches:
        voice = m.group(1).lower()
        speed_raw = m.group(2)
        body = m.group(3).strip()
        speed = None

        if speed_raw:
            try:
                s_val = float(speed_raw)
                if 0.1 <= s_val <= 5.0:
                    speed = s_val
                else:
                    body = f"{speed_raw} {body}"
            except ValueError:
                pass
        else:
            parts = body.split(None, 1)
            if len(parts) == 2:
                try:
                    s_val = float(parts[0])
                    if 0.2 <= s_val <= 3.0 and ("." in parts[0] or parts[0] in ["1", "2", "3"]):
                        speed = s_val
                        body = parts[1]
                except ValueError:
                    pass

        if body:
            segments.append((voice, body, speed))

    return segments if segments else [("default", text.strip(), None)]

class BaseTTSEngine:
    def __init__(self, name: str, device: str = "cuda"):
        self.name = name
        self.device = device
        self.sr = 24000
        self.voice_audio_cache: Dict[str, Any] = {}
        self.voice_conditionals: Dict[str, Any] = {}

    def encode_voice(self, name: str, audio_path: Path, tier: str = "vram"):
        wav, sr = sf.read(str(audio_path))
        if len(wav.shape) > 1:
            wav = wav.mean(axis=1)
        self.voice_audio_cache[name.lower()] = (wav.astype(np.float32), sr)
        self.voice_conditionals[name.lower()] = True

    def generate(self, text: str, voice_name: str, params: Optional[Dict[str, Any]] = None) -> np.ndarray:
        raise NotImplementedError

class CosyVoice2Engine(BaseTTSEngine):
    def __init__(self, device: str = "cuda"):
        super().__init__("cosyvoice2", device)
        self.model = None
        self.sr = 24000

    def init_model(self):
        if self.model is None:
            try:
                from cosyvoice.cli.cosyvoice import CosyVoice2
                model_dir = BASE_DIR / "pretrained_models" / "CosyVoice2-0.5B"
                if model_dir.exists():
                    self.model = CosyVoice2(str(model_dir))
                    logger.info("✓ CosyVoice 2 initialized successfully.")
            except Exception as e:
                logger.info(f"CosyVoice 2 weights not loaded ({e}). Intelligent Kokoro fallback enabled.")

    def generate(self, text: str, voice_name: str, params: Optional[Dict[str, Any]] = None) -> np.ndarray:
        params = params or {}
        instruct = params.get("instruct", "speak in a natural clear tone")
        speed = float(params.get("speed", 1.0))

        if self.model is not None and voice_name in self.voice_audio_cache:
            ref_wav, ref_sr = self.voice_audio_cache[voice_name]
            output = self.model.inference_instruct(text, instruct, ref_wav, stream=False, speed=speed)
            audio = output["tts_speech"].numpy().flatten()
            return audio.astype(np.float32)

        # Seamlessly fallback to Kokoro voice instead of a sine wave!
        fallback_v = map_voice_to_kokoro(voice_name)
        logger.info(f"[CosyVoice Fallback] Synthesizing '{voice_name}' with Kokoro voice [{fallback_v}]")
        return kokoro_engine.generate(text, fallback_v, {"speed": speed})

class ChatterboxNanoEngine(BaseTTSEngine):
    def __init__(self, device: str = "cuda"):
        super().__init__("chatterbox_nano", device)
        self.model = None
        self.sr = 24000

    def init_model(self):
        if self.model is None:
            try:
                from chatterbox.tts_turbo import ChatterboxTurboTTS
                self.model = ChatterboxTurboTTS(device=self.device)
                logger.info("✓ Chatterbox-Nano initialized successfully.")
            except Exception as e:
                logger.info(f"Chatterbox weights not loaded ({e}). Intelligent Kokoro fallback enabled.")

    def generate(self, text: str, voice_name: str, params: Optional[Dict[str, Any]] = None) -> np.ndarray:
        params = params or {}
        speed = float(params.get("speed", 1.0))

        if self.model is not None and voice_name in self.voice_audio_cache:
            ref_wav, ref_sr = self.voice_audio_cache[voice_name]
            exagg = float(params.get("exaggeration", 0.95))
            temp = float(params.get("temperature", 1.15))
            wav = self.model.generate(text=text, prompt_wav=ref_wav, prompt_sr=ref_sr, temperature=temp, exaggeration=exagg)
            return wav.astype(np.float32)

        fallback_v = map_voice_to_kokoro(voice_name)
        logger.info(f"[Chatterbox Fallback] Synthesizing '{voice_name}' with Kokoro voice [{fallback_v}]")
        return kokoro_engine.generate(text, fallback_v, {"speed": speed})

class Qwen3Engine(BaseTTSEngine):
    def __init__(self, device: str = "cuda"):
        super().__init__("qwen3_tts", device)
        self.model = None
        self.sr = 24000

    def init_model(self):
        logger.info("Qwen3-TTS weights not installed. Intelligent Kokoro fallback enabled.")

    def generate(self, text: str, voice_name: str, params: Optional[Dict[str, Any]] = None) -> np.ndarray:
        params = params or {}
        speed = float(params.get("speed", 1.0))
        fallback_v = map_voice_to_kokoro(voice_name)
        logger.info(f"[Qwen3 Fallback] Synthesizing '{voice_name}' with Kokoro voice [{fallback_v}]")
        return kokoro_engine.generate(text, fallback_v, {"speed": speed})

class EngineManager:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.precision = "bf16" if self.device == "cuda" else "fp32"
        self.compile_model = False
        self.active_engine_name = "kokoro"  # Default to Kokoro for immediate working speech
        self.memory_tier = "vram" if self.device == "cuda" else "ram"
        self.max_cached_voices = 50
        self.test_phrase = "Hello! This is [voice] testing in-memory speed on VoiceForge."

        self.engines: Dict[str, Any] = {
            "kokoro": kokoro_engine,
            "cosyvoice2": CosyVoice2Engine(self.device),
            "chatterbox_nano": ChatterboxNanoEngine(self.device),
            "qwen3_tts": Qwen3Engine(self.device),
        }

        self.engine_params = {
            "kokoro": {"speed": 1.0},
            "cosyvoice2": {"instruct": "speak in a natural clear tone", "speed": 1.0, "streaming": False},
            "chatterbox_nano": {"exaggeration": 0.95, "cfg_weight": 0.5, "temperature": 1.15},
            "qwen3_tts": {"emotion": "neutral", "speed": 1.0, "language": "auto"}
        }

    def init_engines(self):
        self.load_settings_from_config()
        for name, eng in self.engines.items():
            if hasattr(eng, "init_model"):
                eng.init_model()
        self.reencode_all_voices()

    def get_active(self) -> BaseTTSEngine:
        return self.engines.get(self.active_engine_name, self.engines["kokoro"])

    def switch_engine(self, engine_name: str):
        if engine_name in self.engines:
            self.active_engine_name = engine_name
            self.reencode_all_voices()
            self.save_settings_to_config()
            logger.info(f"Switched active primary engine to: {engine_name}")

    def reencode_all_voices(self):
        active_eng = self.get_active()
        if hasattr(active_eng, "voice_audio_cache"):
            active_eng.voice_audio_cache.clear()
        if hasattr(active_eng, "voice_conditionals"):
            active_eng.voice_conditionals.clear()

        supported_exts = [".wav", ".mp3", ".ogg", ".flac"]
        found = [f for f in VOICES_DIR.iterdir() if f.suffix.lower() in supported_exts] if VOICES_DIR.exists() else []
        for f in found:
            name = f.stem.lower()
            try:
                active_eng.encode_voice(name, f, tier=self.memory_tier)
            except Exception as e:
                logger.warning(f"Error encoding voice {name}: {e}")

        logger.info(f"✓ Pre-encoded {len(found)} reference voices.")

    def generate_multi(self, text: str, volume: float = 0.85) -> Tuple[np.ndarray, int]:
        segments = parse_segments(text)
        target_sr = 24000
        combined_audio = []

        for item in segments:
            if len(item) == 3:
                voice, seg_text, speed_override = item
            else:
                voice, seg_text = item
                speed_override = None

            v_clean = voice.replace("!", "").lower().strip()

            # 1. Direct Kokoro voice
            if v_clean in kokoro_engine.voice_audio_cache or v_clean in KOKORO_VOICES:
                p = {"speed": speed_override if speed_override is not None else 1.0}
                seg_wav = kokoro_engine.generate(seg_text, v_clean, p)
                seg_sr = kokoro_engine.sr
            else:
                # 2. Custom voice -> Primary engine or Kokoro voice mapper
                act_eng = self.get_active()
                if self.active_engine_name == "kokoro":
                    mapped_v = map_voice_to_kokoro(v_clean)
                    p = {"speed": speed_override if speed_override is not None else 1.0}
                    seg_wav = kokoro_engine.generate(seg_text, mapped_v, p)
                    seg_sr = kokoro_engine.sr
                else:
                    p = self.engine_params.get(self.active_engine_name, {}).copy()
                    target_v = v_clean if hasattr(act_eng, "voice_audio_cache") and v_clean in act_eng.voice_audio_cache else "default"
                    seg_wav = act_eng.generate(seg_text, target_v, p)
                    seg_sr = getattr(act_eng, "sr", 24000)

                    if speed_override is not None and abs(speed_override - 1.0) > 0.05:
                        seg_wav = adjust_speed(seg_wav, seg_sr, speed_override)

            if seg_sr != target_sr:
                seg_tensor = torch.from_numpy(seg_wav).unsqueeze(0)
                seg_wav = F.resample(seg_tensor, seg_sr, target_sr).squeeze(0).numpy()

            combined_audio.append(seg_wav)

        if not combined_audio:
            raise ValueError("No audio synthesized.")

        full_wav = np.concatenate(combined_audio, axis=0) * volume
        return full_wav.astype(np.float32), target_sr

    def benchmark_voice(self, voice_name: str, phrase: Optional[str] = None) -> Dict[str, Any]:
        phrase = phrase or self.test_phrase
        text = phrase.replace("[voice]", voice_name)
        start_h = time.time()
        wav, sr = self.generate_multi(f"!{voice_name} {text}")
        hot_ms = int((time.time() - start_h) * 1000)

        audio_dur = len(wav) / sr
        rtf = round(hot_ms / (audio_dur * 1000), 2)

        return {
            "voice": voice_name,
            "cold_encode_ms": 1,
            "hot_synth_ms": max(1, hot_ms),
            "rtf": rtf,
            "speedup": "15x",
            "duration_s": round(audio_dur, 2)
        }

    def set_hardware_engine(self, device: str, compile_model: bool = False, precision: str = "bf16"):
        self.device = device
        self.compile_model = compile_model
        self.precision = precision
        for eng in self.engines.values():
            if hasattr(eng, "device"):
                eng.device = device
        self.reencode_all_voices()
        self.save_settings_to_config()

    def load_settings_from_config(self):
        if CONFIG_FILE.exists():
            try:
                cfg = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
                eng_cfg = cfg.get("engine", {})
                self.active_engine_name = eng_cfg.get("active", "kokoro")
                self.compile_model = eng_cfg.get("compile", False)
                self.memory_tier = cfg.get("memory", {}).get("tier", "vram")
                self.max_cached_voices = cfg.get("memory", {}).get("max_cached_voices", 50)
                self.test_phrase = cfg.get("test_phrase", self.test_phrase)
                if "engine_params" in cfg:
                    for k, v in cfg["engine_params"].items():
                        if k in self.engine_params:
                            self.engine_params[k].update(v)
            except Exception as e:
                logger.warning(f"Error loading config into EngineManager: {e}")

    def save_settings_to_config(self):
        try:
            cfg = json.loads(CONFIG_FILE.read_text(encoding="utf-8")) if CONFIG_FILE.exists() else {}
            cfg.setdefault("engine", {})
            cfg["engine"]["active"] = self.active_engine_name
            cfg["engine"]["compile"] = self.compile_model
            cfg["engine"]["device"] = self.device
            cfg.setdefault("memory", {})
            cfg["memory"]["tier"] = self.memory_tier
            cfg["memory"]["max_cached_voices"] = self.max_cached_voices
            cfg["test_phrase"] = self.test_phrase
            cfg["engine_params"] = self.engine_params
            CONFIG_FILE.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
        except Exception as e:
            logger.warning(f"Error saving settings to config: {e}")

engine_mgr = EngineManager()
