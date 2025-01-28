# Speech Dataset Pipeline - WIP

- [x] Step 0: Download audio files from RTHK
- [x] Step 1: Split audio files into smaller chunks
- [x] Step 2: Source separation
- [ ] Step 4: Language detection
- [ ] Step 3: Voice enhancement

## Prerequisites

```shell
pip install -r requirements.txt
```

## Usage

```shell
# Download audio file and convert to 16kHz, at this stage, it would create a folder `audios` for original audio files and `audios_16k` for 16kHz audio files
python step-0.py

# Split audio files into smaller chunks and speaker diarization
python step-1.py --audio_root_path audios_16k

# Source separation, remove background music
python step-1.py --audio_root_path chunks

TODO...
```