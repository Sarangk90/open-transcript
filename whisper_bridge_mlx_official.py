#!/usr/bin/env python3
"""
Official MLX Whisper Bridge Script for OpenWhispr
Uses the official mlx-whisper library (60x real-time performance)
Supports pre-quantized models from Hugging Face MLX Community
"""

import sys
import json
import os
import argparse
import shutil
import threading
import time
from pathlib import Path

try:
    import mlx_whisper
    from huggingface_hub import snapshot_download
except ImportError as e:
    print(json.dumps({
        "success": False,
        "error": f"Required libraries not installed: {str(e)}. Run: pip install mlx-whisper huggingface-hub"
    }))
    sys.exit(1)

# Model mapping: UI model name -> HuggingFace repo
MODEL_MAPPING = {
    # Standard models (full precision)
    "tiny": "mlx-community/whisper-tiny",
    "base": "mlx-community/whisper-base",
    "small": "mlx-community/whisper-small",
    "medium": "mlx-community/whisper-medium",
    "large": "mlx-community/whisper-large-v3",
    "large-v3": "mlx-community/whisper-large-v3",

    # Turbo models (4 decoder layers, fastest)
    "turbo": "mlx-community/whisper-large-v3-turbo",
    "large-v3-turbo": "mlx-community/whisper-large-v3-turbo",
    "turbo-q4": "mlx-community/whisper-large-v3-turbo-q4",  # 4-bit quantized

    # 8-bit quantized models (memory efficient)
    "base-8bit": "mlx-community/whisper-base-mlx-8bit",
    "medium-8bit": "mlx-community/whisper-medium-mlx-8bit",
    "large-8bit": "mlx-community/whisper-large-v3-mlx-8bit",
}

def transcribe_audio(audio_file, model_name="base", language=None):
    """Transcribe audio file using official mlx-whisper"""
    import time

    if not os.path.exists(audio_file):
        return {
            "success": False,
            "error": f"Audio file not found: {audio_file}"
        }

    try:
        print(f"[MLX] Starting transcription with model: {model_name}", file=sys.stderr)
        start_time = time.time()

        # Get HuggingFace model repo
        hf_repo = MODEL_MAPPING.get(model_name, "mlx-community/whisper-base")
        print(f"[MLX] Using HuggingFace model: {hf_repo}", file=sys.stderr)

        # Transcribe using mlx-whisper
        transcribe_start = time.time()
        result = mlx_whisper.transcribe(
            audio_file,
            path_or_hf_repo=hf_repo,
            language=language
        )
        transcribe_time = time.time() - transcribe_start

        # Extract text
        text = result.get("text", "").strip()
        if not text or len(text) == 0:
            return {
                "success": False,
                "error": "Transcription produced no text"
            }

        total_time = time.time() - start_time

        # Determine backend type from model name
        if "q4" in model_name or "q4" in hf_repo:
            backend = "mlx-official-q4"
            speedup_note = "4-bit quantized"
        elif "8bit" in model_name or "8bit" in hf_repo:
            backend = "mlx-official-8bit"
            speedup_note = "8-bit quantized"
        elif "turbo" in model_name or "turbo" in hf_repo:
            backend = "mlx-official-turbo"
            speedup_note = "4-layer turbo"
        else:
            backend = "mlx-official"
            speedup_note = "full precision"

        print(f"[MLX] Transcription completed in {transcribe_time:.2f}s (total: {total_time:.2f}s)", file=sys.stderr)
        print(f"[MLX] Official mlx-whisper GPU acceleration ({speedup_note})", file=sys.stderr)

        return {
            "success": True,
            "text": text,
            "language": result.get("language", language or "en"),
            "backend": backend,
            "model_repo": hf_repo,
            "timing": {
                "transcribe_time": transcribe_time,
                "total_time": total_time
            }
        }

    except Exception as e:
        print(f"[MLX] Transcription failed: {str(e)}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
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

def monitor_download_progress(model_name, hf_repo, stop_event):
    """Monitor download progress by watching directory size growth"""
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    repo_folder = f"models--{hf_repo.replace('/', '--')}"
    repo_path = os.path.join(cache_dir, repo_folder)

    # Estimate expected size based on model type
    expected_sizes = {
        "tiny": 80 * 1024 * 1024,
        "base": 150 * 1024 * 1024,
        "small": 500 * 1024 * 1024,
        "medium": 1500 * 1024 * 1024,
        "large": 3000 * 1024 * 1024,
        "turbo": 1600 * 1024 * 1024,
        "turbo-q4": 500 * 1024 * 1024,
        "base-8bit": 100 * 1024 * 1024,
        "medium-8bit": 800 * 1024 * 1024,
        "large-8bit": 1600 * 1024 * 1024,
    }
    expected_size = expected_sizes.get(model_name, 1000 * 1024 * 1024)

    last_size = 0
    last_update_time = time.time()
    speed_samples = []
    last_progress_update = 0

    while not stop_event.is_set():
        try:
            current_size = 0
            if os.path.exists(repo_path):
                # Calculate directory size
                for dirpath, dirnames, filenames in os.walk(repo_path):
                    for filename in filenames:
                        filepath = os.path.join(dirpath, filename)
                        try:
                            current_size += os.path.getsize(filepath)
                        except (OSError, FileNotFoundError):
                            pass

            # Calculate speed
            current_time = time.time()
            time_diff = current_time - last_update_time

            speed_mbps = 0
            if last_size > 0 and time_diff > 0 and current_size > last_size:
                bytes_per_second = (current_size - last_size) / time_diff
                speed_mbps = (bytes_per_second * 8) / (1024 * 1024)

                speed_samples.append(speed_mbps)
                if len(speed_samples) > 10:
                    speed_samples.pop(0)
                speed_mbps = sum(speed_samples) / len(speed_samples)

            percentage = min((current_size / expected_size * 100) if expected_size > 0 else 0, 100)

            # Emit progress every 500ms or if percentage changed significantly
            if (current_time - last_progress_update > 0.5 or
                abs(percentage - last_progress_update) > 1.0):

                progress_data = {
                    "type": "progress",
                    "model": model_name,
                    "downloaded_bytes": current_size,
                    "total_bytes": expected_size,
                    "percentage": round(percentage, 1),
                    "speed_mbps": round(speed_mbps, 2) if speed_mbps > 0 else 0
                }

                print(f"PROGRESS:{json.dumps(progress_data)}", file=sys.stderr, flush=True)
                last_progress_update = percentage

            # Exit if download complete
            if current_size >= expected_size * 0.95 and current_size > 0:
                break

            # Exit if no progress for 30 seconds
            if current_size == last_size and current_time - last_update_time > 30:
                if current_size > expected_size * 0.9:
                    break

            last_size = current_size
            last_update_time = current_time

        except Exception:
            pass

        time.sleep(0.5)

def download_model(model_name="base"):
    """Download MLX Whisper model from HuggingFace with progress tracking"""
    hf_repo = MODEL_MAPPING.get(model_name, "mlx-community/whisper-base")

    stop_event = threading.Event()
    progress_thread = None

    try:
        cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
        repo_folder = f"models--{hf_repo.replace('/', '--')}"
        repo_path = os.path.join(cache_dir, repo_folder)

        # Check if already downloaded
        if os.path.exists(repo_path):
            size_bytes = sum(
                os.path.getsize(os.path.join(dirpath, filename))
                for dirpath, dirnames, filenames in os.walk(repo_path)
                for filename in filenames
            )
            return {
                "model": model_name,
                "downloaded": True,
                "hf_repo": hf_repo,
                "path": repo_path,
                "size_bytes": size_bytes,
                "size_mb": round(size_bytes / (1024 * 1024), 1),
                "success": True
            }

        # Start progress monitoring in background thread
        progress_thread = threading.Thread(
            target=monitor_download_progress,
            args=(model_name, hf_repo, stop_event),
            daemon=True
        )
        progress_thread.start()

        # Download model
        local_path = snapshot_download(
            repo_id=hf_repo,
            cache_dir=cache_dir,
            resume_download=True
        )

        # Stop progress monitoring
        stop_event.set()
        if progress_thread and progress_thread.is_alive():
            progress_thread.join(timeout=1)

        # Get final size
        size_bytes = sum(
            os.path.getsize(os.path.join(dirpath, filename))
            for dirpath, dirnames, filenames in os.walk(local_path)
            for filename in filenames
        )

        # Emit completion
        completion_data = {
            "type": "complete",
            "model": model_name,
            "downloaded_bytes": size_bytes,
            "total_bytes": size_bytes,
            "percentage": 100
        }
        print(f"PROGRESS:{json.dumps(completion_data)}", file=sys.stderr, flush=True)

        return {
            "model": model_name,
            "downloaded": True,
            "hf_repo": hf_repo,
            "path": local_path,
            "size_bytes": size_bytes,
            "size_mb": round(size_bytes / (1024 * 1024), 1),
            "success": True
        }

    except KeyboardInterrupt:
        stop_event.set()
        return {
            "model": model_name,
            "downloaded": False,
            "error": "Download interrupted by user",
            "success": False
        }
    except Exception as e:
        stop_event.set()
        return {
            "model": model_name,
            "downloaded": False,
            "error": str(e),
            "success": False
        }

def check_model_status(model_name="base"):
    """Check if MLX model is downloaded"""
    hf_repo = MODEL_MAPPING.get(model_name, "mlx-community/whisper-base")

    try:
        cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
        repo_folder = f"models--{hf_repo.replace('/', '--')}"
        repo_path = os.path.join(cache_dir, repo_folder)

        if os.path.exists(repo_path):
            size_bytes = sum(
                os.path.getsize(os.path.join(dirpath, filename))
                for dirpath, dirnames, filenames in os.walk(repo_path)
                for filename in filenames
            )

            return {
                "model": model_name,
                "downloaded": True,
                "hf_repo": hf_repo,
                "path": repo_path,
                "size_bytes": size_bytes,
                "size_mb": round(size_bytes / (1024 * 1024), 1),
                "success": True
            }
        else:
            return {
                "model": model_name,
                "downloaded": False,
                "hf_repo": hf_repo,
                "success": True
            }
    except Exception as e:
        return {
            "model": model_name,
            "error": str(e),
            "success": False
        }

def list_models():
    """List all MLX models and their status"""
    models = list(MODEL_MAPPING.keys())
    model_info = []

    for model_name in models:
        status = check_model_status(model_name)
        model_info.append(status)

    return {
        "models": model_info,
        "cache_dir": os.path.expanduser("~/.cache/huggingface/hub"),
        "success": True
    }

def delete_model(model_name="base"):
    """Delete downloaded MLX model"""
    hf_repo = MODEL_MAPPING.get(model_name, "mlx-community/whisper-base")

    try:
        cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
        repo_folder = f"models--{hf_repo.replace('/', '--')}"
        repo_path = os.path.join(cache_dir, repo_folder)

        if os.path.exists(repo_path):
            size_bytes = sum(
                os.path.getsize(os.path.join(dirpath, filename))
                for dirpath, dirnames, filenames in os.walk(repo_path)
                for filename in filenames
            )

            shutil.rmtree(repo_path)

            return {
                "model": model_name,
                "deleted": True,
                "freed_bytes": size_bytes,
                "freed_mb": round(size_bytes / (1024 * 1024), 1),
                "success": True
            }
        else:
            return {
                "model": model_name,
                "deleted": False,
                "error": "Model not found",
                "success": False
            }
    except Exception as e:
        return {
            "model": model_name,
            "deleted": False,
            "error": str(e),
            "success": False
        }

def main():
    parser = argparse.ArgumentParser(description="Official MLX Whisper Bridge with GPU Acceleration")
    parser.add_argument("audio_file", nargs="?", help="Audio file to transcribe")
    parser.add_argument("--model", default="base", help="Model to use (tiny/base/small/medium/large/turbo/turbo-q4/large-8bit)")
    parser.add_argument("--language", default=None, help="Language code")
    parser.add_argument("--output-format", default="json", choices=["json", "text"])
    parser.add_argument("--mode", default="transcribe",
                       choices=["transcribe", "download", "check", "list", "delete", "check-ffmpeg"],
                       help="Operation mode")

    args = parser.parse_args()

    # Handle download mode
    if args.mode == "download":
        result = download_model(args.model)
        print(json.dumps(result))
        sys.exit(0 if result.get("success") else 1)

    # Handle check mode
    elif args.mode == "check":
        result = check_model_status(args.model)
        print(json.dumps(result))
        sys.exit(0 if result.get("success") else 1)

    # Handle list mode
    elif args.mode == "list":
        result = list_models()
        print(json.dumps(result))
        sys.exit(0 if result.get("success") else 1)

    # Handle delete mode
    elif args.mode == "delete":
        result = delete_model(args.model)
        print(json.dumps(result))
        sys.exit(0 if result.get("success") else 1)

    # Handle check-ffmpeg mode
    elif args.mode == "check-ffmpeg":
        result = check_ffmpeg()
        print(json.dumps(result))
        sys.exit(0 if result.get("available") else 1)

    # Handle transcribe mode
    elif args.mode == "transcribe":
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
