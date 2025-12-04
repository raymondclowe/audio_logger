#!/usr/bin/env python3
"""
Interactive A/B tester for sox cleaning chains.
- Records a 10s sample (or uses --input wav)
- Builds a noise profile from the sample
- Produces A and B variants using different sox chains
- Plays A or B via aplay
- Simple TUI: [A] chooses A, [B] chooses B, [R] replays, [S] save transcripts, [Q] quit

Usage:
  python tools/ab_compare.py [--input /path/to.wav] [--device plughw:3,0] [--model base] [--url http://192.168.0.142:8085/transcribe]
"""
import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path
import wave
import json
import time

DEFAULT_URL = "http://192.168.0.142:8085/transcribe"
DEFAULT_DEVICE = "plughw:3,0"

# Safe, incremental chains to try for B, always applied to ORIGINAL (A)
CHAIN_VARIANTS = [
    # Variant 0: TEST 19 baseline - known good
    ["sox", "{src}", "{dst}", "noisered", "{noise}", "0.21", "highpass", "200", "lowpass", "3400", 
     "compand", "0.05,0.2", "6:-70,-60,-40", "-3", "-90", "0.1", 
     "equalizer", "1000", "500", "3", "equalizer", "3000", "1000", "2", 
     "norm", "-3", "rate", "16k", "channels", "1"],
    # Variant 1: Same but less compand
    ["sox", "{src}", "{dst}", "noisered", "{noise}", "0.21", "highpass", "200", "lowpass", "3400", 
     "compand", "0.05,0.2", "6:-70,-65,-40", "-3", "-90", "0.1", 
     "equalizer", "1000", "500", "3", "equalizer", "3000", "1000", "2", 
     "norm", "-3", "rate", "16k", "channels", "1"],
    # Variant 2: Same but different EQ balance
    ["sox", "{src}", "{dst}", "noisered", "{noise}", "0.21", "highpass", "200", "lowpass", "3400", 
     "compand", "0.05,0.2", "6:-70,-60,-40", "-3", "-90", "0.1", 
     "equalizer", "1000", "500", "2", "equalizer", "3000", "1000", "3", 
     "norm", "-3", "rate", "16k", "channels", "1"],
    # Variant 3: Less aggressive noise reduction
    ["sox", "{src}", "{dst}", "noisered", "{noise}", "0.18", "highpass", "200", "lowpass", "3400", 
     "compand", "0.05,0.2", "6:-70,-60,-40", "-3", "-90", "0.1", 
     "equalizer", "1000", "500", "3", "equalizer", "3000", "1000", "2", 
     "norm", "-3", "rate", "16k", "channels", "1"],
    # Variant 4: More aggressive noise reduction
    ["sox", "{src}", "{dst}", "noisered", "{noise}", "0.25", "highpass", "200", "lowpass", "3400", 
     "compand", "0.05,0.2", "6:-70,-60,-40", "-3", "-90", "0.1", 
     "equalizer", "1000", "500", "3", "equalizer", "3000", "1000", "2", 
     "norm", "-3", "rate", "16k", "channels", "1"],
]


def run(cmd):
    return subprocess.run(cmd, check=True)


def record_sample(device: str, dest: Path, secs: int = 10):
    print(f"Recording {secs}s sample from {device} → {dest}")
    cmd = [
        "arecord",
        "-D", device,
        "-d", str(secs),
        "-f", "cd",
        "-t", "wav",
        str(dest)
    ]
    run(cmd)


def build_noise_profile(sample: Path, noise_prof: Path):
    print(f"Building noise profile → {noise_prof}")
    run(["sox", str(sample), "-n", "trim", "0", "0.5", "noiseprof", str(noise_prof)])


def sox_convert_chain(chain: list[str], src: Path, dst: Path, noise_prof: Path) -> list[str]:
    args = [a.format(src=str(src), dst=str(dst), noise=str(noise_prof)) for a in chain]
    run(args)
    return args


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


def make_preview(src: Path, dest: Path, seconds: int = 5):
    try:
        subprocess.run(["sox", str(src), str(dest), "trim", "0", str(seconds)], check=True)
    except subprocess.CalledProcessError:
        import shutil
        shutil.copy(src, dest)

def play(path: Path):
    print(f"Playing: {path}")
    subprocess.run(["aplay", str(path)], check=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', type=Path, help='Optional input WAV to use instead of recording')
    ap.add_argument('--device', default=DEFAULT_DEVICE)
    ap.add_argument('--url', default=DEFAULT_URL)
    ap.add_argument('--model', default='base')
    ap.add_argument('--secs', type=int, default=10)
    args = ap.parse_args()

    workdir = Path(tempfile.mkdtemp(prefix='ab_compare_'))
    sample = workdir / 'sample.wav'
    noise_prof = workdir / 'noise.prof'
    a_wav = workdir / 'A.wav'
    b_wav = workdir / 'B.wav'
    variant_idx = 0

    if args.input and args.input.exists():
        print(f"Using input: {args.input}")
        sample = args.input
    else:
        record_sample(args.device, sample, secs=args.secs)

    try:
        build_noise_profile(sample, noise_prof)
        # A starts as copy of original sample
        import shutil
        shutil.copy(sample, a_wav)
        # B will be built in the loop
    except subprocess.CalledProcessError as e:
        print(f"Error preparing: {e}")
        sys.exit(1)

    # Precompute transcript for A
    a_text = transcribe(args.url, args.model, a_wav)
    b_text = None
    a_cmd = None  # Track sox command that created current A

    # Build preview paths
    a_prev = workdir / 'A_preview.wav'
    b_prev = workdir / 'B_preview.wav'

    print("\nA/B Tester (auto-plays 5s A then 5s B):\n- 1: prefer A (keep A, try next B variant)\n- 2: prefer B (promote B to A, continue)\n- R: replay A then B (5s each)\n- S: show/save transcripts\n- N: skip to next B variant\n- Q: quit")
    current_cmd = None
    last = None
    # Auto-play previews first
    print("\n=== Comparison ===")
    if a_cmd:
        print("A sox command:")
        print(' '.join(a_cmd))
    else:
        print("A: original sample (no processing)")
    print("\nB sox command:")
    current_cmd = sox_convert_chain(CHAIN_VARIANTS[variant_idx], a_wav, b_wav, noise_prof)
    print(' '.join(current_cmd))
    make_preview(a_wav, a_prev, seconds=5)
    make_preview(b_wav, b_prev, seconds=5)
    b_text = transcribe(args.url, args.model, b_wav)
    play(a_prev)
    play(b_prev)
    last = 'B'
    # Continuous loop: promote best-so-far and keep iterating
    while True:
        inp = input("[1/2/R/S/N/Q] > ").strip().upper()
        if inp == '1':
            # Prefer A: keep current A, advance to NEXT B variant
            print("Preferred: A (best-so-far) → trying next B variant")
            if variant_idx + 1 < len(CHAIN_VARIANTS):
                variant_idx += 1
                try:
                    print("\n=== Comparison ===")
                    if a_cmd:
                        print("A sox command:")
                        print(' '.join(a_cmd))
                    else:
                        print("A: original sample (no processing)")
                    print(f"\nB variant #{variant_idx+1} sox command:")
                    current_cmd = sox_convert_chain(CHAIN_VARIANTS[variant_idx], a_wav, b_wav, noise_prof)
                    print(' '.join(current_cmd))
                    b_text = transcribe(args.url, args.model, b_wav)
                    make_preview(b_wav, b_prev, seconds=5)
                    play(a_prev)
                    play(b_prev)
                except subprocess.CalledProcessError as e:
                    print(f"Variant failed (sox): {e}")
            else:
                print("No more variants available. A is your best result.")
                break
            continue
        elif inp == '2':
            # Prefer B: promote B to new A, reset variant index, build next B from new A
            print("Preferred: B → promoting B to new A and continuing")
            import shutil
            shutil.copy(b_wav, a_wav)
            a_cmd = current_cmd  # Save the command that created new A
            variant_idx = 0
            try:
                print("\n=== Comparison ===")
                print("A sox command (promoted from previous B):")
                print(' '.join(a_cmd))
                print(f"\nB variant #{variant_idx+1} sox command:")
                current_cmd = sox_convert_chain(CHAIN_VARIANTS[variant_idx], a_wav, b_wav, noise_prof)
                print(' '.join(current_cmd))
                a_text = transcribe(args.url, args.model, a_wav)
                b_text = transcribe(args.url, args.model, b_wav)
                make_preview(a_wav, a_prev, seconds=5)
                make_preview(b_wav, b_prev, seconds=5)
                play(a_prev)
                play(b_prev)
            except subprocess.CalledProcessError as e:
                print(f"Variant failed (sox) after promotion: {e}")
                continue
        elif inp == 'R':
            play(a_prev)
            play(b_prev)
        elif inp == 'S':
            print("\nTranscript A:\n" + (a_text or '(empty)'))
            print("\nTranscript B:\n" + (b_text or '(empty)'))
            out_json = workdir / 'ab_result.json'
            with open(out_json, 'w') as f:
                json.dump({
                    'A': {'path': str(a_wav), 'transcript': a_text},
                    'B': {
                        'path': str(b_wav),
                        'transcript': b_text,
                        'sox_cmd': current_cmd,
                        'variant_index': variant_idx,
                    },
                    'Original': {'path': str(sample)},
                }, f, indent=2)
            print(f"Saved transcripts: {out_json}")

        elif inp == 'N':
            # Skip to next processing variant (from current A)
            if variant_idx + 1 < len(CHAIN_VARIANTS):
                variant_idx += 1
                try:
                    print("\n=== Comparison ===")
                    if a_cmd:
                        print("A sox command:")
                        print(' '.join(a_cmd))
                    else:
                        print("A: original sample (no processing)")
                    print(f"\nB variant #{variant_idx+1} sox command:")
                    current_cmd = sox_convert_chain(CHAIN_VARIANTS[variant_idx], a_wav, b_wav, noise_prof)
                    print(' '.join(current_cmd))
                    b_text = transcribe(args.url, args.model, b_wav)
                    make_preview(b_wav, b_prev, seconds=5)
                    play(a_prev)
                    play(b_prev)
                except subprocess.CalledProcessError as e:
                    print(f"Variant failed (sox): {e}")
            else:
                print("No more variants available.")
                continue
        elif inp == 'Q':
            break
        else:
            print("Invalid input. Use 1/2/R/S/T/X/N/Q.")

    # Save final state snapshot for traceability
    summary = workdir / 'selection.txt'
    with open(summary, 'w') as f:
        f.write(f"A={a_wav}\nB={b_wav}\nOriginal={sample}\n")
        if a_cmd:
            f.write("A_sox_cmd=")
            f.write(' '.join(a_cmd) + "\n")
        if current_cmd:
            f.write("B_sox_cmd=")
            f.write(' '.join(current_cmd) + "\n")
            f.write(f"B_variant_index={variant_idx}\n")
    print(f"Saved selection snapshot: {summary}")

    print(f"Working directory: {workdir}")

if __name__ == '__main__':
    main()
