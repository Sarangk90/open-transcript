#!/usr/bin/env python3
"""
MLX Whisper Bridge Script for OpenWhispr
Handles GPU-accelerated speech-to-text using Lightning Whisper MLX
Provides identical quality to standard Whisper but with M4 Max GPU acceleration
"""

import sys
import json
import os
import argparse
from pathlib import Path

try:
    import lightning_whisper_mlx
except ImportError:
    print(json.dumps({
        "success": False,
        "error": "Lightning Whisper MLX not installed. Run: pip install lightning-whisper-mlx"
    }))
    sys.exit(1)

# Model name mapping from standard Whisper to MLX
MODEL_MAPPING = {
    "tiny": "tiny",
    "base": "base", 
    "small": "small",
    "medium": "medium",
    "large": "large-v3",
    "large-v3": "large-v3",
    "large-v3-turbo": "large-v3",  # MLX doesn't have turbo yet, use large-v3
    "turbo": "large-v3"
}

_model_cache = {}

def load_model(model_name="base", allow_quantization=True):
    """Load Whisper model with MLX GPU acceleration"""
    global _model_cache

    # Map model name
    mlx_model = MODEL_MAPPING.get(model_name, "base")

    # Return cached model if available
    cache_key = f"{mlx_model}_{'8bit' if allow_quantization else 'full'}"
    if cache_key in _model_cache:
        return _model_cache[cache_key], cache_key

    try:
        quant_param = "8bit" if allow_quantization else None
        print(f"[MLX] Loading model with quantization: {quant_param}", file=sys.stderr)

        model = lightning_whisper_mlx.LightningWhisperMLX(
            model=mlx_model,
            batch_size=12,  # Optimized for M4 Max
            quant=quant_param
        )

        # Cache only one model to save memory
        _model_cache.clear()
        _model_cache[cache_key] = model
        return model, cache_key
    except Exception as e:
        print(f"[MLX] Model load failed: {e}", file=sys.stderr)
        return None, None

def transcribe_audio(audio_file, model_name="base", language=None):
    """Transcribe audio file with MLX GPU acceleration (full precision)"""
    import time

    if not os.path.exists(audio_file):
        return {
            "success": False,
            "error": f"Audio file not found: {audio_file}"
        }

    # Use full precision (8-bit quantization disabled due to library incompatibility)
    try:
        print(f"[MLX] Starting transcription with model: {model_name}", file=sys.stderr)
        start_time = time.time()

        model, cache_key = load_model(model_name, allow_quantization=False)
        if model is None:
            return {
                "success": False,
                "error": f"Failed to load model: {model_name}"
            }

        load_time = time.time() - start_time
        print(f"[MLX] Model loaded in {load_time:.2f}s (full precision)", file=sys.stderr)

        # Transcribe with MLX (GPU accelerated)
        transcribe_start = time.time()
        result = model.transcribe(audio_file, language=language)
        transcribe_time = time.time() - transcribe_start

        # Validate result
        text = result.get("text", "").strip()
        if not text or len(text) == 0:
            return {
                "success": False,
                "error": "Transcription produced no text"
            }

        total_time = time.time() - start_time
        print(f"[MLX] Transcription completed in {transcribe_time:.2f}s (total: {total_time:.2f}s)", file=sys.stderr)
        print(f"[MLX] GPU/Neural Engine acceleration (full precision)", file=sys.stderr)

        return {
            "success": True,
            "text": text,
            "language": result.get("language", "en"),
            "backend": "mlx-gpu",
            "timing": {
                "load_time": load_time,
                "transcribe_time": transcribe_time,
                "total_time": total_time
            }
        }

    except Exception as e:
        return {
            "success": False,
            "error": f"Transcription failed: {str(e)}"
        }

def check_ffmpeg():
    """Check if FFmpeg is available"""
    import subprocess

    # Check for FFmpeg in environment variables first
    ffmpeg_paths = [
        os.environ.get('FFMPEG_PATH'),
        os.environ.get('FFMPEG_EXECUTABLE'),
        os.environ.get('FFMPEG_BINARY'),
        'ffmpeg'  # System ffmpeg
    ]

    for ffmpeg_path in ffmpeg_paths:
        if not ffmpeg_path:
            continue

        try:
            result = subprocess.run(
                [ffmpeg_path, '-version'],
                capture_output=True,
                timeout=5
            )
            if result.returncode == 0:
                return {
                    "available": True,
                    "path": ffmpeg_path
                }
        except (subprocess.TimeoutExpired, FileNotFoundError, PermissionError):
            continue

    return {
        "available": False,
        "error": "FFmpeg not found in PATH or environment variables"
    }

def main():
    parser = argparse.ArgumentParser(description="MLX Whisper Bridge with GPU Acceleration")
    parser.add_argument("audio_file", nargs="?", help="Audio file to transcribe")
    parser.add_argument("--model", default="base", help="Model to use")
    parser.add_argument("--language", default=None, help="Language code")
    parser.add_argument("--output-format", default="json", choices=["json", "text"])
    parser.add_argument("--mode", default="transcribe", choices=["transcribe", "check-ffmpeg"], help="Operation mode")

    args = parser.parse_args()

    # Handle check-ffmpeg mode
    if args.mode == "check-ffmpeg":
        result = check_ffmpeg()
        print(json.dumps(result))
        sys.exit(0 if result.get("available") else 1)

    # Handle transcribe mode
    if not args.audio_file:
        print(json.dumps({
            "success": False,
            "error": "Audio file required"
        }))
        sys.exit(1)

    result = transcribe_audio(args.audio_file, args.model, args.language)

    if args.output_format == "json":
        print(json.dumps(result))
    else:
        if result.get("success"):
            print(result.get("text", ""))
        else:
            print(f"Error: {result.get('error', 'Unknown error')}", file=sys.stderr)
            sys.exit(1)

if __name__ == "__main__":
    main()
