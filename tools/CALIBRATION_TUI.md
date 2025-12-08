# Enhanced Calibration Tool with Rich TUI

## Overview

The `guided_calibration_tui.py` script provides a beautiful, interactive terminal UI for audio calibration with real-time visualization of:

- **Live parameter search progress** - Watch the optimizer explore the N-dimensional parameter space
- **Word-level accuracy comparison** - See exactly which words are transcribed correctly (color-coded)
- **Real-time metrics** - WER, CER, and combined scores update as trials complete
- **Parameter space visualization** - Table showing ranges, baseline, and current best values
- **Interactive progress tracking** - Visual progress bars and trial counters

## Features

### 🎨 Visual Components

1. **Score Display Panel**
   - WER (Word Error Rate) accuracy
   - CER (Character Error Rate) accuracy  
   - Combined score (90% WER + 10% CER)
   - Color-coded: Green for good (>0.8), Yellow for medium

2. **Parameter Search Space Table**
   - Shows all 11 audio processing parameters
   - Range for each parameter
   - Current best value (highlighted when changed from baseline)
   - Baseline comparison

3. **Transcript Comparison Panel**
   - Reference text displayed
   - Best transcript with word-level color coding:
     - **Green**: Correct word
     - **Red**: Wrong/substituted word
     - **Yellow**: Extra word (insertion)
     - **Dim**: Missing word (deletion)

4. **Progress Tracker**
   - Current trial number
   - Visual progress bar
   - Percentage complete
   - Updates in real-time

## Installation

Install the required `rich` library:

```bash
cd /home/pi/audio_logger
uv pip install rich
# or
pip install rich>=13.7.0
```

## Usage

### Basic Usage

```bash
# Auto-detect device, use default settings
python tools/guided_calibration_tui.py

# Specify transcription server
python tools/guided_calibration_tui.py --url http://192.168.0.142:8085

# Use custom text file
python tools/guided_calibration_tui.py --text calibration-text-short.txt

# More trials for better results
python tools/guided_calibration_tui.py --trials 100
```

### Full Options

```bash
python tools/guided_calibration_tui.py \
  --url http://192.168.0.142:8085 \
  --text my_custom_text.txt \
  --device plughw:3,0 \
  --model small \
  --trials 75 \
  --duration 60
```

### Arguments

- `--device DEVICE` - Audio recording device (auto-detected if not specified)
- `--url URL` - Transcription service URL (default: http://localhost:8085/transcribe)
- `--model MODEL` - Whisper model to use (default: small)
- `--trials N` - Number of optimization trials (default: 50)
- `--duration N` - Recording duration in seconds (auto-calculated from text length if not specified)
- `--text FILE` - Path to custom calibration text file

## How It Works

### 1. Recording Phase
- Displays your calibration text
- Records audio with interactive progress bar
- Press ENTER to stop early when done reading
- Auto-calculates recommended recording time based on text length

### 2. Optimization Phase
- Live TUI shows real-time progress
- Bayesian optimization explores parameter space intelligently
- Each trial:
  - Tests a different combination of audio processing parameters
  - Transcribes the processed audio
  - Calculates WER and CER scores
  - Updates UI if this combination is better
- **Composite scoring**: 90% WER + 10% CER (WER dominates, CER breaks ties)

### 3. Results
- Best parameters automatically saved to `room_calibration_config.json`
- Main audio logger will use these settings on next run
- Full results history saved for analysis

## Comparison: Basic vs TUI Version

### Basic Version (`guided_calibration.py`)
```
[1/50] WER: 0.950, CER: 0.962, Combined: 0.951
[2/50] WER: 0.900, CER: 0.925, Combined: 0.902
...
```
- Simple text output
- Minimal feedback
- Hard to track progress

### TUI Version (`guided_calibration_tui.py`)
```
┌─────────────────────────────────────────┐
│     🎯 AUDIO CALIBRATION OPTIMIZER      │
├─────────────────────────────────────────┤
│                                         │
│  📊 Best Scores                         │
│  WER: 0.950  CER: 0.962  Combined: 0.951│
│                                         │
│  🔍 Parameter Search Space              │
│  ┌──────────┬──────────┬──────┬────────┐│
│  │Parameter │ Range    │ Best │Baseline││
│  │noisered  │ 0.1-0.35 │ 0.19 │ 0.21   ││
│  ...                                    │
│                                         │
│  📝 Transcript Comparison               │
│  This is a very short warrior calibr... │
│  (word-level color highlighting)        │
│                                         │
│  ⏳ Progress: [████████░░] 45/50 90%   │
└─────────────────────────────────────────┘
```
- Rich visual interface
- Real-time updates
- Word-level accuracy display
- Parameter tracking
- Progress visualization

## Tips for Best Results

1. **Use longer text** (50-100+ words) to better differentiate parameters
2. **Include variety** - different words, sentence structures, punctuation
3. **Background noise** - Some ambient noise helps test noise reduction
4. **More trials** - 75-100 trials give better optimization results
5. **Normal conditions** - Record in your typical usage environment

## Example Text Files

### Short Test (20 words)
```
This is a very short audio calibration test.
By adding an additional paragraph we can make it a better test.
```

### Recommended Length (50+ words)
Use the default `room_calibration-example.txt` which contains ~200 words from Darwin's Beagle voyage.

## Troubleshooting

### TUI not displaying correctly
- Ensure terminal supports Unicode and colors
- Try a different terminal emulator
- Check terminal size (minimum 80x24 recommended)

### Recording issues
- Check microphone connection: `arecord -l`
- Test recording: `arecord -D plughw:3,0 -d 5 test.wav`
- Verify device parameter is correct

### All scores are 1.000 (perfect)
- Text too simple/short - use longer, more complex text
- Audio quality too good - test with some background noise
- See warning messages in output for recommendations

## Output Files

- `room_calibration_config.json` - Best parameters (used by main audio logger)
- `calibration_output/guided_calibration_sample.wav` - Recorded audio
- `calibration_output/noise.prof` - Noise profile
- `calibration_output/trial_NNNN.wav` - Processed audio for each trial

## Technical Details

### Parameter Space (11 dimensions)
- `noisered`: 0.1-0.35 (noise reduction amount)
- `highpass`: 50-500 Hz (high-pass filter)
- `lowpass`: 2500-4500 Hz (low-pass filter)
- `compand_attack`: 0.03-0.15 (compressor attack)
- `compand_decay`: 0.15-0.40 (compressor decay)
- `eq1_freq`: 700-900 Hz (EQ band 1 frequency)
- `eq1_width`: 300-500 Hz (EQ band 1 width)
- `eq1_gain`: 3-6 dB (EQ band 1 gain)
- `eq2_freq`: 2300-3200 Hz (EQ band 2 frequency)
- `eq2_width`: 800-1200 Hz (EQ band 2 width)
- `eq2_gain`: 1-4 dB (EQ band 2 gain)

### Scoring Algorithm
- **WER (Word Error Rate)**: Measures word-level accuracy (insertions, deletions, substitutions)
- **CER (Character Error Rate)**: Measures character-level accuracy (captures punctuation)
- **Combined**: `0.9 × WER + 0.1 × CER` ensures WER is primary, CER breaks ties

### Optimization Strategy
Uses Bayesian optimization (TPE sampler from Optuna):
- Learns from previous trials
- Focuses on promising parameter regions
- More efficient than random/grid search
- Typically converges in 50-100 trials

## See Also

- `guided_calibration.py` - Basic version without TUI
- `room_calibration.py` - Advanced tool with more options
- `README.md` - Main project documentation
