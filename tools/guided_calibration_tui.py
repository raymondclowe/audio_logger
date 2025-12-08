#!/usr/bin/env python3
"""
Enhanced Guided Room Calibration with Rich TUI.

This version provides a beautiful terminal UI with:
- Live parameter search visualization
- Word-level accuracy comparison with color coding
- Real-time progress tracking
- N-dimensional parameter space visualization
- Interactive display of best results

Usage:
    python tools/guided_calibration_tui.py
    python tools/guided_calibration_tui.py --url http://192.168.0.142:8085
    python tools/guided_calibration_tui.py --text my_text.txt --trials 100
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass
import threading
import select

from rich.console import Console
from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
from rich.table import Table
from rich.text import Text
from rich import box
from rich.align import Align
from rich.columns import Columns

# Import from room_calibration module
from room_calibration import (
    BASELINE_PARAMS,
    CALIBRATION_CONFIG_FILE,
    DEFAULT_MODEL,
    DEFAULT_URL,
    build_noise_profile,
    calculate_word_accuracy,
    run_bayesian_optimization,
    PARAMETER_SPACE,
    HAS_OPTUNA,
    HAS_JIWER,
    CalibrationResult,
)

console = Console()


@dataclass
class CalibrationState:
    """Track calibration progress state."""
    current_trial: int = 0
    total_trials: int = 50
    best_wer: float = 0.0
    best_cer: float = 0.0
    best_combined: float = 0.0
    best_params: Optional[Dict[str, str]] = None
    current_params: Optional[Dict[str, str]] = None  # Parameters being tested in current trial
    best_transcript: str = ""
    reference_text: str = ""
    all_results: Optional[List[CalibrationResult]] = None
    param_ranges: Optional[Dict[str, Tuple[float, float, float]]] = None  # (min, max, step)
    original_ranges: Optional[Dict[str, Tuple[float, float, float]]] = None  # Track original ranges to detect expansions
    
    def __post_init__(self):
        if self.best_params is None:
            self.best_params = {}
        if self.current_params is None:
            self.current_params = {}
        if self.all_results is None:
            self.all_results = []
        if self.param_ranges is None:
            self.param_ranges = {}


def detect_audio_device() -> str:
    """Auto-detect audio recording device."""
    try:
        from guided_calibration import detect_audio_device as detect
        return detect()
    except:
        return "plughw:3,0"


def load_sample_text(text_file: Optional[Path] = None) -> str:
    """Load sample text from file."""
    try:
        from guided_calibration import load_sample_text as load_text
        return load_text(text_file)
    except:
        if text_file and text_file.exists():
            return text_file.read_text().strip()
        return "This is a fallback calibration test."


def calculate_recording_duration(text: str, words_per_minute: int = 150) -> int:
    """Calculate recording duration based on text length."""
    word_count = len(text.split())
    base_seconds = int((word_count / words_per_minute) * 60)
    return base_seconds + 30


def record_audio_16k_mono(device: str, output_path: Path, duration: int = 60) -> bool:
    """Record audio with 3-second countdown for ambient noise capture."""
    # Constants for recording timing
    RECORDING_START_DELAY = 0.2  # Seconds to wait for recording process to start
    COUNTDOWN_DURATION = 3  # Seconds of silence to capture for noise profiling
    
    console.print(f"\n[yellow]🎤 Starting recording...[/yellow]")
    
    cmd = [
        "arecord",
        "-D", device,
        "-f", "S16_LE",
        "-r", "16000",
        "-c", "1",
        "-t", "wav",
        str(output_path)
    ]
    
    try:
        # Start recording BEFORE the countdown to capture ambient noise
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Brief delay to ensure recording has started
        time.sleep(RECORDING_START_DELAY)
        
        console.print(f"[dim]   Stay SILENT during countdown (capturing ambient noise)...[/dim]")
        
        # Countdown while recording captures silence
        for i in range(COUNTDOWN_DURATION, 0, -1):
            console.print(f"[bold cyan]   {i}...[/bold cyan]")
            time.sleep(1)
        
        console.print(f"[bold green]   🔴 NOW READING[/bold green] (up to {duration} seconds)")
        console.print("[dim]   Press ENTER when done reading to stop early[/dim]\n")
        
        stop_recording = threading.Event()
        
        def wait_for_enter():
            try:
                if select.select([sys.stdin], [], [], 0)[0]:
                    sys.stdin.readline()
                input()
                stop_recording.set()
            except:
                pass
        
        enter_thread = threading.Thread(target=wait_for_enter, daemon=True)
        enter_thread.start()
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console
        ) as progress:
            task = progress.add_task("Recording...", total=duration)
            
            # Account for seconds already elapsed during countdown
            progress.update(task, advance=COUNTDOWN_DURATION)
            
            # Continue recording for remaining duration
            for elapsed in range(COUNTDOWN_DURATION, duration):
                if stop_recording.is_set() or process.poll() is not None:
                    break
                time.sleep(1)
                progress.update(task, advance=1)
        
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        
        if output_path.exists() and output_path.stat().st_size > 1000:
            console.print("[green]✓ Recording complete![/green]\n")
            return True
        else:
            console.print("[red]✗ Recording failed[/red]\n")
            return False
            
    except Exception as e:
        console.print(f"[red]✗ Recording error: {e}[/red]\n")
        if process.poll() is None:
            process.kill()
        return False


def create_word_comparison(reference: str, transcript: str) -> Text:
    """Create color-coded word-by-word comparison showing original transcript with punctuation."""
    if not HAS_JIWER:
        return Text(transcript)
    
    import jiwer
    
    # Show the original transcript with punctuation/capitalization from Whisper
    # Split by whitespace to preserve punctuation attached to words
    trans_words = transcript.split()
    
    # For comparison, normalize reference to match word content (not punctuation)
    transforms = jiwer.Compose([
        jiwer.ToLowerCase(),
        jiwer.RemovePunctuation(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip(),
    ])
    ref_words = transforms(reference).split()
    
    # For comparison logic, normalize transcript words too (remove punctuation for matching)
    trans_words_normalized = [transforms(word) for word in trans_words]
    
    result = Text()
    trans_idx = 0
    ref_idx = 0
    
    # Simple word-by-word comparison showing original words with punctuation
    while trans_idx < len(trans_words):
        trans_word_original = trans_words[trans_idx]
        trans_word_norm = trans_words_normalized[trans_idx]
        
        if ref_idx < len(ref_words) and ref_words[ref_idx] == trans_word_norm:
            # Correct word (show original with punctuation)
            result.append(trans_word_original + " ", style="green bold")
            ref_idx += 1
            trans_idx += 1
        elif ref_idx < len(ref_words):
            # Substitution or wrong word (show original with punctuation)
            result.append(trans_word_original + " ", style="red bold")
            ref_idx += 1
            trans_idx += 1
        else:
            # Extra words at end (show original with punctuation)
            result.append(f"+{trans_word_original}+ ", style="yellow bold")
            trans_idx += 1
    
    # Show missing words from reference
    while ref_idx < len(ref_words):
        result.append(f"[{ref_words[ref_idx]}] ", style="red dim")
        ref_idx += 1
    
    return result


def create_parameter_table(state: CalibrationState) -> Table:
    """Create visualization of parameter search space."""
    table = Table(title="🔍 Parameter Search Space", box=box.ROUNDED, show_header=True)
    
    table.add_column("Parameter", style="cyan", width=15)
    table.add_column("Range", style="white", width=15)
    table.add_column("Current", style="yellow", width=10)
    table.add_column("Best", style="green bold", width=10)
    table.add_column("Baseline", style="dim", width=10)
    
    if not state.param_ranges:
        return table
    
    for param_name, (min_val, max_val, _) in state.param_ranges.items():
        current_val = state.current_params.get(param_name, "—") if state.current_params else "—"
        best_val = state.best_params.get(param_name, "—") if state.best_params else "—"
        baseline_val = BASELINE_PARAMS.get(param_name, "—")
        
        # Check if range expanded (compare to original if available)
        range_expanded = False
        if state.original_ranges and param_name in state.original_ranges:
            orig_min, orig_max, _ = state.original_ranges[param_name]
            if min_val < orig_min or max_val > orig_max:
                range_expanded = True
        
        range_str = f"{min_val:.2f}–{max_val:.2f}"
        
        # Highlight expanded ranges
        if range_expanded:
            range_style = "yellow bold"
        else:
            range_style = "white"
        
        # Highlight if best changed from baseline
        if best_val != baseline_val:
            best_style = "green bold"
        else:
            best_style = "white"
        
        table.add_row(
            param_name,
            Text(range_str, style=range_style),
            str(current_val) if isinstance(current_val, str) else f"{current_val:.2f}",
            str(best_val) if isinstance(best_val, str) else f"{best_val:.2f}",
            str(baseline_val) if isinstance(baseline_val, str) else f"{baseline_val:.2f}"
        )
    
    return table


def create_score_display(state: CalibrationState) -> Panel:
    """Create score metrics display."""
    if state.current_trial == 0:
        content = Text("Waiting for first trial...", style="dim italic")
    else:
        wer_text = Text()
        wer_text.append("WER: ", style="bold")
        wer_text.append(f"{state.best_wer:.3f}", style="green bold" if state.best_wer > 0.8 else "yellow")
        
        cer_text = Text()
        cer_text.append("CER: ", style="bold")
        cer_text.append(f"{state.best_cer:.3f}", style="green bold" if state.best_cer > 0.8 else "yellow")
        
        combined_text = Text()
        combined_text.append("Combined: ", style="bold")
        combined_text.append(f"{state.best_combined:.3f}", style="cyan bold")
        
        content = Columns([wer_text, cer_text, combined_text], equal=True, expand=True)
    
    return Panel(
        content,
        title="📊 Best Scores",
        border_style="cyan",
        box=box.DOUBLE
    )


def create_transcript_comparison(state: CalibrationState) -> Panel:
    """Create transcript comparison panel."""
    if not state.best_transcript:
        content = Text("No transcripts yet...", style="dim italic")
    else:
        content = Text()
        content.append("Reference: ", style="bold cyan")
        content.append(state.reference_text[:80] + "...\n\n", style="white")
        content.append("Best Transcript:\n", style="bold green")
        
        comparison = create_word_comparison(state.reference_text, state.best_transcript)
        content.append(comparison)
    
    return Panel(
        content,
        title="📝 Transcript Comparison",
        subtitle="[green]✓ Correct[/green] [red]✗ Wrong[/red] [yellow]+Extra+[/yellow] [dim][Missing][/dim]",
        border_style="blue",
        box=box.ROUNDED
    )


def create_progress_display(state: CalibrationState) -> Panel:
    """Create progress bar display."""
    if state.total_trials == 0:
        percent = 0
    else:
        percent = (state.current_trial / state.total_trials) * 100
    
    # Create progress bar
    bar_width = 40
    filled = int(bar_width * state.current_trial / state.total_trials) if state.total_trials > 0 else 0
    bar = "█" * filled + "░" * (bar_width - filled)
    
    text = Text()
    text.append(f"Trial {state.current_trial}/{state.total_trials} ", style="bold")
    text.append(f"[{bar}] ", style="cyan")
    text.append(f"{percent:.1f}%", style="green bold")
    
    return Panel(
        Align.center(text),
        title="⏳ Optimization Progress",
        border_style="green",
        box=box.HEAVY
    )


def create_layout(state: CalibrationState) -> Layout:
    """Create the main TUI layout."""
    layout = Layout()
    
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="body"),
        Layout(name="footer", size=3)
    )
    
    layout["body"].split_row(
        Layout(name="left"),
        Layout(name="right")
    )
    
    layout["left"].split_column(
        Layout(name="scores", size=5),
        Layout(name="params")
    )
    
    layout["right"].split_column(
        Layout(name="transcript")
    )
    
    # Update panels
    layout["header"].update(
        Panel(
            Align.center(Text("🎯 AUDIO CALIBRATION OPTIMIZER", style="bold magenta")),
            border_style="magenta"
        )
    )
    
    layout["scores"].update(create_score_display(state))
    layout["params"].update(create_parameter_table(state))
    layout["transcript"].update(create_transcript_comparison(state))
    layout["footer"].update(create_progress_display(state))
    
    return layout


def run_optimization_with_tui(
    audio_path: Path,
    sample_text: str,
    url: str,
    model: str,
    trials: int,
    noise_profile: Path,
    workdir: Path,
    adaptive_ranges: bool = False
) -> List[CalibrationResult]:
    """Run optimization with live TUI updates."""
    
    # Initialize state
    from room_calibration import PARAMETER_RANGES
    import copy
    
    state = CalibrationState(
        total_trials=trials,
        reference_text=sample_text,
        param_ranges=copy.deepcopy(PARAMETER_RANGES),
        original_ranges=copy.deepcopy(PARAMETER_RANGES)  # Track original for expansion detection
    )
    
    results = []
    
    # Custom callback to update TUI
    def create_objective_with_ui(live: Live):
        def objective_callback(trial_num: int, result: CalibrationResult):
            state.current_trial = trial_num
            results.append(result)
            
            # Update current parameters being tested
            state.current_params = result.params.copy()
            
            # Detect range expansions by checking if current params exceed known ranges
            if adaptive_ranges and state.param_ranges:
                for param_name, param_str_value in result.params.items():
                    if param_name not in state.param_ranges:
                        continue
                    
                    try:
                        param_value = float(param_str_value)
                    except ValueError:
                        continue
                    
                    min_val, max_val, step = state.param_ranges[param_name]
                    
                    # If param value exceeds current range, update the range
                    # (this indicates the optimization expanded the range)
                    if param_value < min_val:
                        state.param_ranges[param_name] = (param_value, max_val, step)
                    elif param_value > max_val:
                        state.param_ranges[param_name] = (min_val, param_value, step)
            
            # Update best scores
            combined = 0.9 * result.accuracy_score + 0.1 * result.cer_score
            if combined > state.best_combined:
                state.best_wer = result.accuracy_score
                state.best_cer = result.cer_score
                state.best_combined = combined
                state.best_params = result.params.copy()
                state.best_transcript = result.transcript
            
            state.all_results = results
            
            # Update display
            live.update(create_layout(state))
        
        return objective_callback
    
    with Live(create_layout(state), console=console, refresh_per_second=4) as live:
        callback = create_objective_with_ui(live)
        
        # Run optimization with callback
        results = run_bayesian_optimization(
            input_audio=audio_path,
            reference_text=sample_text,
            url=url,
            model=model,
            noise_profile=noise_profile,
            workdir=workdir,
            n_trials=trials,
            metric="wer",
            verbose=False,
            progress_callback=callback,
            adaptive_ranges=adaptive_ranges
        )
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Enhanced guided calibration with rich TUI visualization."
    )
    
    parser.add_argument("--device", help="Audio recording device")
    parser.add_argument("--url", default=DEFAULT_URL, help=f"Transcription service URL (default: {DEFAULT_URL})")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Whisper model (default: {DEFAULT_MODEL})")
    parser.add_argument("--trials", type=int, default=50, help="Number of optimization trials (default: 50)")
    parser.add_argument("--adaptive-ranges", action="store_true", help="Automatically expand parameter ranges when boundaries are hit (runs in 3 phases)")
    parser.add_argument("--duration", type=int, help="Recording duration in seconds (auto-calculated if not specified)")
    parser.add_argument("--text", help="Path to custom text file")
    
    args = parser.parse_args()
    
    # Auto-detect device
    device = args.device or detect_audio_device()
    
    # Normalize URL
    url = args.url
    if not url.endswith('/transcribe'):
        url = url.rstrip('/') + '/transcribe'
    
    # Load text
    text_file = Path(args.text) if args.text else None
    sample_text = load_sample_text(text_file)
    
    # Calculate duration
    recording_duration = args.duration or calculate_recording_duration(sample_text)
    
    # Show initial info
    console.print(Panel.fit(
        f"[cyan]Device:[/cyan] {device}\n"
        f"[cyan]URL:[/cyan] {url}\n"
        f"[cyan]Model:[/cyan] {args.model}\n"
        f"[cyan]Trials:[/cyan] {args.trials}\n"
        f"[cyan]Text length:[/cyan] {len(sample_text.split())} words\n"
        f"[cyan]Recording time:[/cyan] {recording_duration}s",
        title="⚙️  Configuration",
        border_style="cyan"
    ))
    
    console.print(f"\n[bold yellow]📖 Please read this text clearly:[/bold yellow]")
    console.print(f"[yellow]{sample_text}[/yellow]\n")
    
    console.print("[dim]When recording starts:[/dim]")
    console.print("[dim]  • Stay silent during the 3-second countdown (for noise profiling)[/dim]")
    console.print("[dim]  • Then read the text above clearly[/dim]")
    console.print("[dim]  • Press ENTER when finished to stop early[/dim]\n")
    
    input("Press ENTER to start recording...")
    
    # Create working directory
    workdir = Path(__file__).parent / "calibration_output"
    workdir.mkdir(exist_ok=True)
    
    # Record audio
    audio_path = workdir / "guided_calibration_sample.wav"
    if not record_audio_16k_mono(device, audio_path, recording_duration):
        console.print("[red]✗ Recording failed[/red]")
        sys.exit(1)
    
    # Build noise profile
    console.print("[cyan]📊 Building noise profile from ambient sound...[/cyan]")
    noise_profile = workdir / "noise.prof"
    if not build_noise_profile(audio_path, noise_profile):
        console.print("[red]✗ Failed to build noise profile[/red]")
        sys.exit(1)
    
    console.print("\n[bold green]🔍 Starting optimization with live visualization...[/bold green]\n")
    time.sleep(1)
    
    # Run optimization with TUI
    try:
        results = run_optimization_with_tui(
            audio_path=audio_path,
            sample_text=sample_text,
            url=url,
            model=args.model,
            trials=args.trials,
            noise_profile=noise_profile,
            workdir=workdir,
            adaptive_ranges=args.adaptive_ranges
        )
    except KeyboardInterrupt:
        console.print("\n[yellow]Calibration cancelled[/yellow]")
        sys.exit(1)
    
    if not results:
        console.print("[red]✗ No results produced[/red]")
        sys.exit(1)
    
    # Sort and display final results
    results.sort(key=lambda r: (0.9 * r.accuracy_score + 0.1 * r.cer_score), reverse=True)
    best = results[0]
    
    # Save configuration
    results_data = {
        "best_params": best.params,
        "best_wer_score": best.accuracy_score,
        "best_cer_score": best.cer_score,
        "best_transcript": best.transcript,
        "sox_command": best.sox_command,
        "reference_text": sample_text[:100] + "...",
        "calibration_device": device,
        "all_results": [
            {
                "params": r.params,
                "wer_score": r.accuracy_score,
                "cer_score": r.cer_score,
                "transcript": r.transcript
            }
            for r in results[:10]
        ]
    }
    
    with open(CALIBRATION_CONFIG_FILE, 'w') as f:
        json.dump(results_data, f, indent=2)
    
    console.print(Panel.fit(
        f"[green]✓ Configuration saved to:[/green] {CALIBRATION_CONFIG_FILE}\n\n"
        "[cyan]The main transcriber will automatically use these optimized settings.[/cyan]",
        title="✅ Calibration Complete!",
        border_style="green"
    ))


if __name__ == "__main__":
    main()
