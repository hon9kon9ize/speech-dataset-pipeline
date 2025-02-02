import os
import glob
import argparse
from tqdm.auto import tqdm
import pandas as pd
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess

model_dir = "iic/SenseVoiceSmall"

model = AutoModel(
    model=model_dir,
    device="cuda:0",
    batch_size=64,
)

languages = ["zn", "en", "yue", "ja", "ko", "nospeech"]


def transcribe(audio_path: str):
    res = model.generate(
        input=audio_path,
        cache={},
        language="auto",  # "zn", "en", "yue", "ja", "ko", "nospeech"
        use_itn=True,
    )
    batch_results = []

    for r in res:
        key = r["key"]
        lang = [f"<|{lang}|>" in r["text"] for lang in languages][0]
        batch_results.append((key, lang, rich_transcription_postprocess(r["text"])))

    return batch_results


def main(root_folder: str, batch_size=64):
    model.kwargs["batch_size"] = batch_size

    mp3_files = glob.glob(os.path.join(root_folder, "*.mp3")) + glob.glob(
        os.path.join(root_folder, "**/*.mp3"), recursive=True
    )

    os.makedirs("transcriptions", exist_ok=True)

    results = []

    print("Total audio files:", len(mp3_files))
    print("Transcribing...")

    for i in range(0, len(mp3_files), batch_size):
        batch_files = mp3_files[i : i + batch_size]

        res = transcribe(batch_files)
        results += res

    df = pd.DataFrame(results, columns=["key", "lang", "text"])
    df.to_csv("transcriptions/sensevoice.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_root_path", required=True, type=str)
    parser.add_argument("--batch_size", default=64, type=int)
    args = parser.parse_args()

    main(args.audio_root_path, args.batch_size)
