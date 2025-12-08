#!/usr/bin/env python3
"""
Demo of the TUI components for the guided calibration tool.
Shows what the interface looks like without running a full calibration.
"""

from pathlib import Path
import time
from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.live import Live
from rich import box
from rich.align import Align

console = Console()

# Simulate calibration state
class DemoState:
    def __init__(self):
        self.current_trial = 0
        self.total_trials = 50
        self.best_wer = 0.0
        self.best_cer = 0.0
        self.best_combined = 0.0
        self.reference_text = "This is a very short audio calibration test by adding an additional paragraph we can make it a better test"
        self.best_transcript = ""
        self.best_params = {
            "noisered": "0.19",
            "highpass": "500",
            "lowpass": "4000",
            "compand_attack": "0.09",
            "compand_decay": "0.15",
            "eq1_freq": "700",
            "eq1_width": "300",
            "eq1_gain": "6",
            "eq2_freq": "3200",
            "eq2_width": "1000",
            "eq2_gain": "1",
        }

def create_demo_layout(state):
    """Create demo layout."""
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
    
    # Header
    layout["header"].update(
        Panel(
            Align.center(Text("🎯 AUDIO CALIBRATION OPTIMIZER", style="bold magenta")),
            border_style="magenta"
        )
    )
    
    # Scores
    if state.current_trial > 0:
        from rich.columns import Columns
        wer_text = Text()
        wer_text.append("WER: ", style="bold")
        wer_text.append(f"{state.best_wer:.3f}", style="green bold")
        
        cer_text = Text()
        cer_text.append("CER: ", style="bold")
        cer_text.append(f"{state.best_cer:.3f}", style="green bold")
        
        combined_text = Text()
        combined_text.append("Combined: ", style="bold")
        combined_text.append(f"{state.best_combined:.3f}", style="cyan bold")
        
        scores_content = Columns([wer_text, cer_text, combined_text], equal=True, expand=True)
    else:
        scores_content = Text("Starting optimization...", style="dim italic")
    
    layout["scores"].update(
        Panel(scores_content, title="📊 Best Scores", border_style="cyan", box=box.DOUBLE)
    )
    
    # Parameters table
    param_table = Table(title="🔍 Parameter Search Space", box=box.ROUNDED, show_header=True)
    param_table.add_column("Parameter", style="cyan", width=15)
    param_table.add_column("Range", style="white", width=15)
    param_table.add_column("Best", style="green bold", width=10)
    param_table.add_column("Baseline", style="dim", width=10)
    
    params_display = [
        ("noisered", "0.1–0.35", "0.19", "0.21"),
        ("highpass", "50–500", "500", "300"),
        ("lowpass", "2500–4500", "4000", "3400"),
        ("compand_attack", "0.03–0.15", "0.09", "0.03"),
        ("eq1_freq", "700–900", "700", "800"),
        ("eq2_gain", "1–4", "1", "3"),
    ]
    
    for param, range_str, best, baseline in params_display:
        param_table.add_row(param, range_str, best, baseline)
    
    layout["params"].update(param_table)
    
    # Transcript comparison
    if state.best_transcript:
        transcript_text = Text()
        transcript_text.append("Reference: ", style="bold cyan")
        transcript_text.append(state.reference_text[:60] + "...\n\n", style="white")
        transcript_text.append("Best Transcript:\n", style="bold green")
        
        # Color-coded words
        words = [
            ("This", "green"), ("is", "green"), ("a", "green"), ("very", "green"),
            ("short", "green"), ("warrior", "red"), ("calibration", "green"),
            ("test", "green"), ("by", "green"), ("adding", "green"), ("an", "green"),
            ("additional", "green"), ("paragraph", "green"), ("we", "green"),
            ("can", "green"), ("make", "green"), ("it", "green"), ("a", "green"),
            ("better", "green"), ("test", "green"),
        ]
        
        for word, color in words:
            if color == "green":
                transcript_text.append(word + " ", style="green bold")
            elif color == "red":
                transcript_text.append(word + " ", style="red bold")
        
        transcript_content = transcript_text
    else:
        transcript_content = Text("Waiting for results...", style="dim italic")
    
    layout["transcript"].update(
        Panel(
            transcript_content,
            title="📝 Transcript Comparison",
            subtitle="[green]✓ Correct[/green] [red]✗ Wrong[/red] [yellow]+Extra+[/yellow] [dim][Missing][/dim]",
            border_style="blue",
            box=box.ROUNDED
        )
    )
    
    # Progress
    percent = (state.current_trial / state.total_trials) * 100 if state.total_trials > 0 else 0
    bar_width = 40
    filled = int(bar_width * state.current_trial / state.total_trials) if state.total_trials > 0 else 0
    bar = "█" * filled + "░" * (bar_width - filled)
    
    progress_text = Text()
    progress_text.append(f"Trial {state.current_trial}/{state.total_trials} ", style="bold")
    progress_text.append(f"[{bar}] ", style="cyan")
    progress_text.append(f"{percent:.1f}%", style="green bold")
    
    layout["footer"].update(
        Panel(
            Align.center(progress_text),
            title="⏳ Optimization Progress",
            border_style="green",
            box=box.HEAVY
        )
    )
    
    return layout

def main():
    console.print("\n[bold cyan]🎨 TUI Calibration Interface Demo[/bold cyan]\n")
    console.print("This shows what the interface looks like during optimization.\n")
    
    state = DemoState()
    
    # Simulate trials
    trials_data = [
        (5, 0.550, 0.679, "This is a very short warning of calibration tests..."),
        (10, 0.750, 0.811, "This is a very short warning and haderation test..."),
        (15, 0.850, 0.934, "This is a very short order of calibration test..."),
        (25, 0.900, 0.925, "This is a very short warrior calibration test..."),
        (35, 0.950, 0.962, "This is a very short warrior calibration test by adding..."),
        (50, 0.950, 0.962, "This is a very short warrior calibration test by adding an additional paragraph we can make it a better test"),
    ]
    
    with Live(create_demo_layout(state), console=console, refresh_per_second=4) as live:
        for trial, wer, cer, transcript in trials_data:
            # Simulate progress to this trial
            while state.current_trial < trial:
                state.current_trial += 1
                time.sleep(0.05)
                live.update(create_demo_layout(state))
            
            # Update best scores
            state.best_wer = wer
            state.best_cer = cer
            state.best_combined = 0.9 * wer + 0.1 * cer
            state.best_transcript = transcript
            
            live.update(create_demo_layout(state))
            time.sleep(1.5)
    
    console.print("\n[bold green]✅ Demo complete![/bold green]")
    console.print("\nRun the actual tool with:")
    console.print("  [cyan]python tools/guided_calibration_tui.py --url http://192.168.0.142:8085[/cyan]\n")

if __name__ == "__main__":
    main()
