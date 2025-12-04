#!/usr/bin/env python3
"""
A/B harness for sox cleaning settings with human-in-the-loop.
- Takes an input wav
- Produces variants (A/B/C...) using different sox chains
- Computes simple audio stats (RMS, peak)
- Transcribes each via existing Mini Transcriber
- Heuristically scores transcripts for gibberish
- Prints a summary and provides shell commands to play files

Usage:
  python tools/ab_test_sox.py /path/to/input.wav --model base --url http://192.168.0.142:8085/transcribe

"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Dict
import wave
import numpy as np

DEFAULT_URL = "http://192.168.0.142:8085/transcribe"

VARIANTS = [
    {
        "name": "v1_baseline",
        "chain": [
            ["sox", "{src}", "{dst}", "noisered", "{noise}", "0.21", "highpass", "200", "treble", "6", "norm", "-3"]
        ]
    },
    {
        "name": "v2_stronger_nr",
        "chain": [
            ["sox", "{src}", "{dst}", "noisered", "{noise}", "0.25", "highpass", "200", "treble", "6", "norm", "-3"]
        ]
    },
    {
        "name": "v3_rate16k_mono",
        "chain": [
            ["sox", "{src}", "{dst}", "rate", "16k", "channels", "1", "noisered", "{noise}", "0.21", "highpass", "200", "treble", "6", "norm", "-3"]
        ]
    },
    {
        "name": "v4_rate16k_mono_stronger",
        "chain": [
            ["sox", "{src}", "{dst}", "rate", "16k", "channels", "1", "noisered", "{noise}", "0.25", "highpass", "200", "treble", "6", "norm", "-3"]
        ]
    },
]


def build_noise_profile(input_wav: Path, noise_prof: Path) -> None:
    subprocess.run(["sox", str(input_wav), "-n", "trim", "0", "0.5", "noiseprof", str(noise_prof)], check=True)


def run_chain(chain: List[List[str]], src: Path, dst: Path, noise_prof: Path) -> None:
    flat: List[str] = []
    for part in chain:
        flat.extend(part)
    # Python format identifiers cannot be 'in'/'out' (keywords); use 'src'/'dst'
    fmt = {
        'src': str(src),
        'dst': str(dst),
        'noise': str(noise_prof),
    }
    cmd = [arg.format(**fmt) for arg in flat]
    subprocess.run(cmd, check=True)


def wav_stats(path: Path) -> Dict[str, float]:
    with wave.open(str(path), 'rb') as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        n_frames = wf.getnframes()
        frames = wf.readframes(n_frames)
    dtype = np.int16 if sampwidth == 2 else np.int16
    audio = np.frombuffer(frames, dtype=dtype)
    if n_channels > 1:
        audio = audio.reshape(-1, n_channels).mean(axis=1).astype(np.int16)
    if audio.size == 0:
        return {"rms": 0.0, "peak": 0.0}
    rms = float(np.sqrt(np.mean((audio.astype(np.float64))**2)))
    peak = float(np.max(np.abs(audio)))
    return {"rms": rms, "peak": peak, "rate": framerate, "channels": n_channels}


def transcribe(url: str, model: str, wav_path: Path) -> str:
    import requests
    with open(wav_path, 'rb') as f:
        files = {'file': f}
        params = {'model': model}
        try:
            r = requests.post(url, files=files, params=params, timeout=60)
            r.raise_for_status()
            return r.json().get('text', '').strip()
        except Exception as e:
            return f"ERROR: {e}"


def gibberish_score(text: str) -> float:
    if not text:
        return 1.0
    total = len(text)
    if total == 0:
        return 1.0
    alnum = sum(c.isalnum() or c.isspace() for c in text)
    punct = sum(c in ",.;:!?-'\"" for c in text)
    bad_ratio = 1.0 - ((alnum + punct) / total)
    # Longer is generally better; penalize ultra-short results
    length_penalty = 1.0 if len(text) < 10 else 0.0
    return min(1.0, bad_ratio + length_penalty)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('input_wav', type=Path)
    ap.add_argument('--url', default=DEFAULT_URL)
    ap.add_argument('--model', default='base')
    ap.add_argument('--outdir', default=None)
    args = ap.parse_args()

    src = args.input_wav
    if not src.exists():
        print(f"Input not found: {src}")
        sys.exit(1)

    outdir = Path(args.outdir) if args.outdir else Path(tempfile.mkdtemp(prefix='ab_sox_'))
    outdir.mkdir(exist_ok=True, parents=True)
    noise_prof = outdir / 'noise.prof'
    build_noise_profile(src, noise_prof)

    results = []
    for v in VARIANTS:
        dst = outdir / f"{v['name']}.wav"
        try:
            run_chain(v['chain'], src, dst, noise_prof)
            stats = wav_stats(dst)
            text = transcribe(args.url, args.model, dst)
            score = gibberish_score(text)
            results.append({
                'name': v['name'],
                'path': str(dst),
                'stats': stats,
                'transcript': text,
                'gibberish_score': score,
            })
        except subprocess.CalledProcessError as e:
            results.append({
                'name': v['name'],
                'path': str(dst),
                'error': f"sox failed: {e}",
            })

    # Print summary
    print("\nA/B Summary (lower gibberish_score is better):")
    for r in results:
        if 'error' in r:
            print(f"- {r['name']}: ERROR {r['error']}")
        else:
            s = r['stats']
            print(f"- {r['name']}: rms={s['rms']:.1f} peak={s['peak']:.1f} rate={s['rate']}ch={s['channels']} gib={r['gibberish_score']:.3f}")
            print(f"  transcript: {r['transcript'][:120]}")
            print(f"  play: aplay '{r['path']}'")

    # Save JSON for later review
    out_json = outdir / 'results.json'
    with open(out_json, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nDetails saved to: {out_json}")

if __name__ == '__main__':
    main()
