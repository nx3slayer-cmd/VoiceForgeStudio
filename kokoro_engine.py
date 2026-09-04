import re
import json
import torch
import numpy as np
import soundfile as sf
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List

BASE_DIR = Path(__file__).resolve().parent
CONFIG_FILE = BASE_DIR / "config.json"

# Language definition catalog
KOKORO_LANGUAGES = {
    "en_us": {"code": "a", "name": "American English", "flag": "🇺🇸", "default": True, "prefixes": ["af", "am"]},
    "en_gb": {"code": "b", "name": "British English", "flag": "🇬🇧", "default": True, "prefixes": ["bf", "bm"]},
    "es":    {"code": "e", "name": "Spanish", "flag": "🇪🇸", "default": False, "prefixes": ["ef", "em"]},
    "fr":    {"code": "f", "name": "French", "flag": "🇫🇷", "default": False, "prefixes": ["ff", "fm"]},
    "it":    {"code": "i", "name": "Italian", "flag": "🇮🇹", "default": False, "prefixes": ["if", "im"]},
    "ja":    {"code": "j", "name": "Japanese", "flag": "🇯🇵", "default": False, "prefixes": ["jf", "jm"]},
    "pt":    {"code": "p", "name": "Portuguese", "flag": "🇧🇷", "default": False, "prefixes": ["pf", "pm"]},
    "hi":    {"code": "h", "name": "Hindi", "flag": "🇮🇳", "default": False, "prefixes": ["hf", "hm"]},
    "zh":    {"code": "z", "name": "Mandarin", "flag": "🇨🇳", "default": False, "prefixes": ["zf", "zm"]},
}

# Complete 54-voice catalog with language associations
ALL_KOKORO_VOICES = {
    # American English Female (af)
    "heart": ("af_heart", "en_us"), "bella": ("af_bella", "en_us"), "nicole": ("af_nicole", "en_us"),
    "aoede": ("af_aoede", "en_us"), "kore": ("af_kore", "en_us"), "sarah": ("af_sarah", "en_us"),
    "nova": ("af_nova", "en_us"), "sky": ("af_sky", "en_us"), "alloy": ("af_alloy", "en_us"),
    "jessica": ("af_jessica", "en_us"), "river": ("af_river", "en_us"),
    # American English Male (am)
    "michael": ("am_michael", "en_us"), "adam": ("am_adam", "en_us"), "echo": ("am_echo", "en_us"),
    "eric": ("am_eric", "en_us"), "liam": ("am_liam", "en_us"), "onyx": ("am_onyx", "en_us"),
    "puck": ("am_puck", "en_us"), "fenrir": ("am_fenrir", "en_us"),
    # British English Female (bf)
    "emma": ("bf_emma", "en_gb"), "isabella": ("bf_isabella", "en_gb"), "alice": ("bf_alice", "en_gb"),
    "lily": ("bf_lily", "en_gb"),
    # British English Male (bm)
    "george": ("bm_george", "en_gb"), "fable": ("bm_fable", "en_gb"), "lewis": ("bm_lewis", "en_gb"),
    "daniel": ("bm_daniel", "en_gb"),
    # Spanish (ef / em)
    "dora": ("ef_dora", "es"), "alex": ("em_alex", "es"), "santa": ("em_santa", "es"),
    # French (ff)
    "siwis": ("ff_siwis", "fr"),
    # Italian (if / im)
    "sara": ("if_sara", "it"), "nicola": ("im_nicola", "it"),
    # Japanese (jf / jm)
    "nezumi": ("jf_nezumi", "ja"), "tebukuro": ("jf_tebukuro", "ja"),
    "gongitsune": ("jf_gongitsune", "ja"), "kumo": ("jm_kumo", "ja"),
    # Portuguese (pf / pm)
    "portuguese_dora": ("pf_dora", "pt"), "portuguese_alex": ("pm_alex", "pt"),
    # Hindi (hf / hm)
    "hindi_alpha": ("hf_alpha", "hi"), "hindi_beta": ("hf_beta", "hi"),
    "omega": ("hm_omega", "hi"), "psi": ("hm_psi", "hi"),
    # Mandarin (zf / zm)
    "xiaobei": ("zf_xiaobei", "zh"), "xiaoni": ("zf_xiaoni", "zh"),
    "xiaoxiao": ("zf_xiaoxiao", "zh"), "xiaoyi": ("zf_xiaoyi", "zh"),
    "yunjian": ("zm_yunjian", "zh"), "yunxi": ("zm_yunxi", "zh"),
    "yunxia": ("zm_yunxia", "zh"), "yunyang": ("zm_yunyang", "zh")
}

KOKORO_VOICES = {name: full_id for name, (full_id, _) in ALL_KOKORO_VOICES.items()}
for full_id, _ in ALL_KOKORO_VOICES.values():
    KOKORO_VOICES[full_id] = full_id

class KokoroEngine:
    def __init__(self, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.sr = 24000
        self.pipelines: Dict[str, Any] = {}
        self.enabled_languages: Dict[str, bool] = self.load_language_config()
        self.voice_audio_cache: Dict[str, str] = {}
        self.voice_conditionals: Dict[str, bool] = {}
        self.refresh_enabled_voices()

    def load_language_config(self) -> Dict[str, bool]:
        defaults = {lang: info["default"] for lang, info in KOKORO_LANGUAGES.items()}
        if CONFIG_FILE.exists():
            try:
                cfg = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
                saved = cfg.get("kokoro_languages", {})
                return {**defaults, **saved}
            except Exception:
                return defaults
        return defaults

    def save_language_config(self, new_config: Dict[str, bool]):
        self.enabled_languages = {**self.enabled_languages, **new_config}
        try:
            cfg = json.loads(CONFIG_FILE.read_text(encoding="utf-8")) if CONFIG_FILE.exists() else {}
            cfg["kokoro_languages"] = self.enabled_languages
            CONFIG_FILE.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
        except Exception:
            pass
        self.refresh_enabled_voices()

    def refresh_enabled_voices(self):
        """Updates active voices so only enabled languages are callable or listed."""
        self.voice_audio_cache.clear()
        self.voice_conditionals.clear()

        for name, (full_id, lang_key) in ALL_KOKORO_VOICES.items():
            if self.enabled_languages.get(lang_key, False):
                self.voice_audio_cache[name.lower()] = full_id
                self.voice_audio_cache[full_id.lower()] = full_id
                self.voice_conditionals[name.lower()] = True
                self.voice_conditionals[full_id.lower()] = True

    def _get_pipeline(self, lang_code: str):
        if lang_code not in self.pipelines:
            from kokoro import KPipeline
            self.pipelines[lang_code] = KPipeline(lang_code=lang_code, device=self.device)
        return self.pipelines[lang_code]

    def resolve_voice_and_lang(self, voice_name: str) -> Tuple[str, str]:
        v_clean = voice_name.replace("!", "").lower().strip()
        full_voice = KOKORO_VOICES.get(v_clean, "af_heart")
        
        prefix = full_voice[:2]
        lang_map = {
            "af": "a", "am": "a", "bf": "b", "bm": "b",
            "ef": "e", "em": "e", "ff": "f", "fm": "f",
            "hf": "h", "hm": "h", "if": "i", "im": "i",
            "jf": "j", "jm": "j", "pf": "p", "pm": "p",
            "zf": "z", "zm": "z",
        }
        lang_code = lang_map.get(prefix, "a")
        return full_voice, lang_code

    def encode_voice(self, name: str, audio_path=None, tier: str = "vram"):
        self.voice_audio_cache[name.lower()] = name
        self.voice_conditionals[name.lower()] = True

    def generate(self, text: str, voice_name: str, params: Optional[Dict[str, Any]] = None) -> np.ndarray:
        params = params or {}
        speed = float(params.get("speed", 1.0))
        full_voice, lang_code = self.resolve_voice_and_lang(voice_name)

        pipeline = self._get_pipeline(lang_code)
        generator = pipeline(text, voice=full_voice, speed=speed)

        audio_pieces = []
        for _, _, audio in generator:
            if audio is not None and len(audio) > 0:
                audio_pieces.append(audio)

        if not audio_pieces:
            raise ValueError(f"Kokoro synthesized 0 audio samples for '{text}' using voice '{voice_name}'.")

        wav_np = np.concatenate(audio_pieces, axis=0).astype(np.float32)
        return wav_np

kokoro_engine = KokoroEngine()
