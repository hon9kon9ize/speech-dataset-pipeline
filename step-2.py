import argparse
import glob
import os
import torch
import tempfile
from tqdm.auto import tqdm
import numpy as np
import librosa
import multiprocessing as mp
import subprocess
import soundfile as sf

device = "cuda" if torch.cuda.is_available() else "cpu"

temp_dir = tempfile.gettempdir()


def separate_audio(mp3_path: str):
    mp3_filename = os.path.basename(mp3_path)
    audio_folder = os.path.dirname(mp3_path)
    file_name_without_extension = os.path.splitext(mp3_filename)[0]
    tmp_vocal_path = os.path.join(
        temp_dir, "mdx_extra", f"{file_name_without_extension}/vocals.wav"
    )
    tmp_bgm_path = os.path.join(
        temp_dir, "mdx_extra", f"{file_name_without_extension}/no_vocals.wav"
    )
    vocals_path = os.path.join(
        audio_folder, file_name_without_extension + "_vocals.mp3"
    )

    if file_name_without_extension.endswith("_vocals"):
        pass
    else:
        command = [
            "demucs",
            "--two-stems",
            "vocals",
            mp3_path,
            "-o",
            temp_dir,
            "-n",
            "mdx_extra",
        ]
        subprocess.run(
            command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        vocal, _ = librosa.load(tmp_vocal_path, sr=16_000)
        bgm, _ = librosa.load(tmp_bgm_path, sr=16_000)
        snr = 10 * np.log10(np.mean(vocal**2) / np.mean(bgm**2))

        if snr.item() > 4:
            sf.write(
                vocals_path,
                vocal,
                16_000,
                format="mp3",
            )

        # remove the temporary files
        os.remove(tmp_vocal_path)
        os.remove(tmp_bgm_path)


def main(root_folder, num_proc=8):
    mp3_files = glob.glob(os.path.join(root_folder, "**/*.mp3"))

    with mp.Pool(processes=num_proc) as pool:
        list(tqdm(pool.imap_unordered(separate_audio, mp3_files), total=len(mp3_files)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_root_path", required=True, type=str)
    parser.add_argument("--num_proc", type=int, default=8)
    args = parser.parse_args()

    main(args.audio_root_path, num_proc=args.num_proc)

#  python step-2.py --audio_root_path "xxx"
