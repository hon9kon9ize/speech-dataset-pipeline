import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

import argparse
import os
import multiprocessing
import soundfile as sf
import torchaudio
from tqdm.auto import tqdm
from diarizer import Diarizer

diar = Diarizer(
    embed_model="ecapa",  # supported types: ['xvec', 'ecapa']
    cluster_method="sc",  # supported types: ['ahc', 'sc']
    window=1.5,  # size of window to extract embeddings (in seconds)
    period=0.75,  # hop of window (in seconds)
)


def vad(
    path_to_audiofile: str,
    out_dir="chunks",
    sampling_rate=16_000,
):
    audio_dir = os.path.splitext(os.path.basename(path_to_audiofile))[0]
    wav, _ = torchaudio.load(path_to_audiofile)
    segments = diar.diarize(wav, sample_rate=sampling_rate)

    os.makedirs(f"{out_dir}/{audio_dir}", exist_ok=True)

    for i, segment in enumerate(segments):
        start = int(segment["start"] * 16000)
        end = int(segment["end"] * 16000)
        label = f"speaker{segment['label']:02d}"
        chunk = wav[0, start:end]

        sf.write(
            f"{out_dir}/{audio_dir}/{i:03d}-{start}-{end}-{label}.mp3",
            chunk,
            16000,
            format="mp3",
        )


def vad_split(root_folder, num_proc=8):
    os.makedirs("chunks", exist_ok=True)

    audio_files = os.listdir(root_folder)
    output_files = [
        os.path.join(root_folder, audio_file)
        for audio_file in audio_files
        if audio_file.endswith(".mp3")
    ]

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
