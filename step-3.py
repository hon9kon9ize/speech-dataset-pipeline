import os
import argparse
import multiprocessing as mp
import glob
from tqdm.auto import tqdm
import soundfile as sf
from speechbrain.inference.separation import SepformerSeparation as separator

model = separator.from_hparams(
    source="speechbrain/sepformer-dns4-16k-enhancement",
    savedir="pretrained_models/sepformer-dns4-16k-enhancement",
)


def enhance_audio(path_to_audiofile: str):
    est_sources = model.separate_file(path_to_audiofile)
    wav = est_sources.view(1, -1)[0].detach().cpu().numpy()

    audio_folder = os.path.dirname(path_to_audiofile).split("/")[-1]

    os.makedirs(f"enhanced/{audio_folder}", exist_ok=True)

    sf.write(
        f"enhanced/{audio_folder}/{os.path.basename(path_to_audiofile)}",
        wav,
        16000,
        format="mp3",
    )


def main(root_folder: str, num_proc=8):
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass

    os.makedirs("enhanced", exist_ok=True)

    audio_files = glob.glob(
        os.path.join(root_folder, "**/*.mp3"), recursive=True
    ) + glob.glob(os.path.join(root_folder, "*.mp3"))
    output_files = [
        audio_file for audio_file in audio_files if audio_file.endswith(".mp3")
    ]

    print("Audio files:", len(audio_files))

    with mp.Pool(processes=num_proc) as pool:
        list(tqdm(pool.imap(enhance_audio, output_files), total=len(output_files)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_root_path", required=True, type=str)
    parser.add_argument("--num_proc", type=int, default=8)
    args = parser.parse_args()

    main(args.audio_root_path, args.num_proc)
