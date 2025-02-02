import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

import argparse
import os
import multiprocessing
import soundfile as sf
import torchaudio
import torch
from tqdm.auto import tqdm

model, utils = torch.hub.load(
    repo_or_dir="snakers4/silero-vad",
    model="silero_vad",
    onnx=True,
    force_reload=False,
)
SAMPLING_RATE = 16000
(get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils
vad_iterator = VADIterator(model, sampling_rate=SAMPLING_RATE)


def vad(
    path_to_audiofile: str,
    out_dir="chunks",
    sampling_rate=16_000,
):
    try:
        audio_dir = os.path.splitext(os.path.basename(path_to_audiofile))[0]
        wav, _ = torchaudio.load(path_to_audiofile)
        speech_timestamps = get_speech_timestamps(
            wav,
            model,
            sampling_rate=sampling_rate,
        )

        os.makedirs(f"{out_dir}/{audio_dir}", exist_ok=True)

        for i, segment in enumerate(speech_timestamps):
            start = int(segment["start"])
            end = int(segment["end"])
            chunk = wav[0, start:end]

            sf.write(
                f"{out_dir}/{audio_dir}/{i:03d}-{start}-{end}.mp3",
                chunk,
                16000,
                format="mp3",
            )
    except:
        print("Failed to separate", path_to_audiofile)


def vad_split(root_folder, num_proc=8):
    os.makedirs("chunks", exist_ok=True)

    audio_files = os.listdir(root_folder)
    output_files = [
        os.path.join(root_folder, audio_file)
        for audio_file in audio_files
        if audio_file.endswith(".mp3")
    ]

    print("Audio files:", len(audio_files))

    with multiprocessing.Pool(processes=num_proc) as pool:
        list(tqdm(pool.imap_unordered(vad, output_files), total=len(output_files)))


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")

    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_root_path", required=True, type=str)
    parser.add_argument("--num_proc", type=int, default=8)
    args = parser.parse_args()

    vad_split(args.audio_root_path, num_proc=args.num_proc)


# python step-1.py --audio_root_path "xx"
