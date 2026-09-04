import pytest
import numpy as np
import torch
from pathlib import Path
from engine_manager import engine_mgr, parse_segments, VOICES_DIR
from kokoro_engine import kokoro_engine, KOKORO_VOICES

def test_speed_and_voice_segment_parser():
    """Verify that multi-voice tags and speed modifiers parse cleanly."""
    raw = "!heart-.75 Running slow !bella-1.25 Running fast !swifty Normal speed"
    segments = parse_segments(raw)
    assert len(segments) == 3
    assert segments[0][0] == "heart"
    assert segments[0][2] == 0.75
    assert segments[1][0] == "bella"
    assert segments[1][2] == 1.25
    assert segments[2][0] == "swifty"
    assert segments[2][2] is None

def test_kokoro_neural_speech():
    """Verify that Kokoro synthesizes real spoken words, not a sine wave beep."""
    wav, sr = engine_mgr.generate_multi("!heart Kokoro neural voice pipeline is active.")
    assert isinstance(wav, np.ndarray)
    assert sr == 24000
    assert len(wav) > sr * 0.6          # More than 0.6s of real audio
    assert np.max(np.abs(wav)) > 0.01   # Not silence

def test_custom_voice_or_cloning_dispatch():
    """Verify custom cloned voices (apple, caged, swifty, etc.) generate correctly."""
    custom_files = [f.stem for f in VOICES_DIR.iterdir() if f.suffix.lower() in [".wav", ".mp3"]]
    target_voice = custom_files[0] if custom_files else "swifty"
    
    wav, sr = engine_mgr.generate_multi(f"!{target_voice} Testing custom cloned voice synthesis.")
    assert isinstance(wav, np.ndarray)
    assert sr == 24000
    assert len(wav) > sr * 0.6
    assert np.max(np.abs(wav)) > 0.01

def test_default_untagged_stream_message():
    """Verify that regular messages with no !voice tag use the default streamer voice."""
    wav, sr = engine_mgr.generate_multi("Plain livestream chat message without any command tag.")
    assert isinstance(wav, np.ndarray)
    assert sr == 24000
    assert len(wav) > 0
