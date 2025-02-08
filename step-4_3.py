import os
import glob
import argparse
from tqdm.auto import tqdm
import pandas as pd
import librosa
import torch
import multiprocessing as mp
from ngram_processor import NgramLogitsProcessor
from transformers import pipeline

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
start_token_id = 50363
ngram_processor = NgramLogitsProcessor(
    "words.txt_correct.arpa",
    0.25,
    lm_start_token_id=start_token_id,
    top_k=10,
)
pipe = pipeline(
    task="automatic-speech-recognition",
    model="alvanlii/whisper-small-cantonese",
    chunk_length_s=30,
    device=device,
    torch_dtype=torch_dtype,
    generate_kwargs={
        "temperature": 0.5,
        "logits_processor": [ngram_processor],
    },
    return_timestamps=False,
)
pipe.model.config.forced_decoder_ids = pipe.tokenizer.get_decoder_prompt_ids(
    language="zh", task="transcribe"
)
pipe.model.generation_config.suppress_tokens = None


def transcribe(mp3_files):
    global ngram_processor

    # clear cache
    ngram_processor.clear_score_cache()

    audio_batch = get_input_features(mp3_files)
    outputs = pipe(
        audio_batch["audio"],
    )
    audio_batch["transcription"] = [o["text"] for o in outputs]

    # remove audio in batch
    audio_batch.pop("audio")

    return audio_batch


def get_input_features(mp3_files):
    output_batch = {
        "audio_file": [],
        "audio": [],
    }

    for mp3_file in mp3_files:
        try:
            wav = librosa.load(mp3_file, sr=16_000)[0]
            duration = wav.shape[0] // 16_000
            if (duration >= 5) or (
                duration <= 30
            ):  # below 30 seconds or above 5 seconds
                output_batch["audio_file"].append(mp3_file)
                output_batch["audio"].append(wav)
        except Exception as e:
            print(f"Error processing {mp3_file}: {e}")

    return output_batch


def main(
    root_folder: str,
    batch_size=64,
    num_proc=8,
):
    mp3_files = glob.glob(os.path.join(root_folder, "*.mp3")) + glob.glob(
        os.path.join(root_folder, "**/*.mp3"), recursive=True
    )
    os.makedirs("transcriptions", exist_ok=True)

    print("Total audio files:", len(mp3_files))

    results = {"audio_file": [], "transcription": []}

    pipe.batch_size = batch_size

    # resume from last checkpoint
    if os.path.exists("transcriptions/whisper-v2.csv"):
        df = pd.read_csv("transcriptions/whisper-v2.csv")
        results = {
            "audio_file": df["audio_file"].tolist(),
            "transcription": df["transcription"].tolist(),
        }
        mp3_files = [
            mp3_file for mp3_file in mp3_files if mp3_file not in results["audio_file"]
        ]

    print("Transcribing...")

    with mp.Pool(num_proc) as pool:
        for output in tqdm(
            pool.imap(
                transcribe,
                [
                    mp3_files[i : i + batch_size]
                    for i in range(0, len(mp3_files), batch_size)
                ],
            ),
            total=len(mp3_files) // batch_size,
        ):
            results["audio_file"].extend(output["audio_file"])
            results["transcription"].extend(output["transcription"])

            if (
                len(results["audio_file"]) % 10_000 == 0
                and len(results["audio_file"]) > 0
            ):
                df = pd.DataFrame(results)
                df = df[["audio_file", "transcription"]]
                df.to_csv("transcriptions/whisper-v2.csv", index=False)

    df = pd.DataFrame(results)
    df = df[["audio_file", "transcription"]]
    df.to_csv("transcriptions/whisper-v2.csv", index=False)


if __name__ == "__main__":
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass

    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_root_path", required=True, type=str)
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--num_proc", default=8, type=int)
    args = parser.parse_args()

    main(args.audio_root_path, args.batch_size, args.num_proc)
