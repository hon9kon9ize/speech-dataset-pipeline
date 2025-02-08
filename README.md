# Speech Dataset Pipeline - WIP

- [x] Step 0: Download audio files from RTHK
- [x] Step 1: Split audio files into smaller chunks
- [x] Step 2: Source separation
- [x] Step 3: Voice enhancement
- [x] Step 4: Transcribe audio files
  - [x] Step 4.1: Transcribe audio files using SenseVoiceSmall with LID
  - [x] Step 4.2: Transcribe audio files using Whisper V3
  - [x] Step 4.23: Transcribe audio files using Cantonese Whisper V2
- [ ] Step 5: Transcription Post-processing

## Prerequisites

```shell
pip install -r requirements.txt
```

## Usage

```shell
# Download audio file and convert to 16kHz, at this stage, it would create a folder `audios` for original audio files and `audios_16k` for 16kHz audio files
python step-0.py

# Source separation, remove background music
python step-1.py --audio_root_path audios_16k

# Split audio files into smaller chunks
python step-2.py --audio_root_path vocals

# Voice enhancement
python step-3.py --audio_root_path enhanced

# Transcribe audio files using SenseVoiceSmall with LID
python step-4_1.py --audio_root_path enhanced

# Transcribe audio files using Whisper V3
python step-4_2.py --audio_root_path enhanced

# Transcribe audio files using Cantonese Whisper V2
python step-4_3.py --audio_root_path enhanced
```