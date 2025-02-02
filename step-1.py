import argparse
import glob
import os
from tqdm.auto import tqdm
import numpy as np
import librosa
import logging
import multiprocessing as mp
import soundfile as sf
from audio_separator.separator import Separator

separator = Separator(
    use_autocast=True,
    mdxc_params={
        "batch_size": 128,
    },
    output_dir="tmp",
    log_level=logging.ERROR,
)

# Load a model
separator.load_model(model_filename="melband_roformer_instvox_duality_v2.ckpt")


def separate_audio(mp3_path: str):
    mp3_filename = os.path.basename(mp3_path)
    vocals_path = os.path.join("vocals", mp3_filename)

    if os.path.exists(vocals_path):
        return

    try:
        tmp_bgm_path, tmp_vocal_path = separator.separate(mp3_path)
        tmp_bgm_path = os.path.join("tmp", tmp_bgm_path)
        tmp_vocal_path = os.path.join("tmp", tmp_vocal_path)

        wav, _ = librosa.load(tmp_vocal_path, sr=16000, mono=True)
        sf.write(vocals_path, wav, 16000)

        # remove the temporary files
        os.remove(tmp_bgm_path)
    except:
        pass


def main(root_folder, num_proc=8):
    mp3_files = glob.glob(os.path.join(root_folder, "*.mp3"))

    # create the target folder
    os.makedirs("vocals", exist_ok=True)
    # create a tmp folder
    os.makedirs("tmp", exist_ok=True)

    with mp.Pool(processes=num_proc) as pool:
        list(tqdm(pool.imap(separate_audio, mp3_files), total=len(mp3_files)))


if __name__ == "__main__":
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass

    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_root_path", required=True, type=str)
    parser.add_argument("--num_proc", type=int, default=8)
    args = parser.parse_args()

    main(args.audio_root_path, num_proc=args.num_proc)

#  python step-2.py --audio_root_path "xxx"
