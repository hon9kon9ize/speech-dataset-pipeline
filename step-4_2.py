import os
import glob
import argparse
from tqdm.auto import tqdm
import pandas as pd
from dataclasses import dataclass
from torch.utils.data import DataLoader
from typing import Any, Dict, List, Union
import numpy as np
import librosa
import pickle
import torch
from datasets import Dataset
from transformers import (
    EncoderDecoderCache,
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
)

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model = (
    AutoModelForSpeechSeq2Seq.from_pretrained(
        "openai/whisper-large-v3",
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
        attn_implementation="sdpa",
    )
    .eval()
    .to(device)
)

processor = AutoProcessor.from_pretrained("openai/whisper-large-v3")

prompt_text, prompt_wav = pickle.load(open("prompt.pkl", "rb"))
blank_token = processor.tokenizer.encode(" ")[0]


def encoder_forward(input_features):
    with torch.no_grad():
        output_attentions = model.model.config.output_attentions
        output_hidden_states = model.model.config.output_hidden_states
        encoder_outputs = model.model.encoder(
            input_features,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

    return encoder_outputs, output_attentions, output_hidden_states


def decoder_forward(
    decoder_input_ids, encoder_outputs, past_key_values=None, use_cache=False
):
    with torch.no_grad():
        outputs = model.model.decoder(
            input_ids=decoder_input_ids,
            past_key_values=past_key_values,
            encoder_hidden_states=encoder_outputs[0],
            use_cache=use_cache,
            return_dict=True,
        )

    return outputs


def prompt_generation(
    batch,
    **kwargs,
):
    batch_size = batch["input_features"].shape[0]
    forced_decoder_ids = processor.get_decoder_prompt_ids(
        language="yue", task="transcribe"
    )
    decoder_input_ids = [model.config.decoder_start_token_id] + [
        f[1] for f in forced_decoder_ids
    ]
    encoder_outputs, output_attentions, output_hidden_states = encoder_forward(
        batch["input_features"]
    )
    past_key_values = None
    decoder_input_ids = torch.LongTensor(
        [decoder_input_ids + prompt_text + [blank_token]] * batch_size
    ).to(device)
    outputs = decoder_forward(
        decoder_input_ids[:, :-1],
        encoder_outputs,
        past_key_values=EncoderDecoderCache.from_legacy_cache(past_key_values),
        use_cache=True,
    )
    past_key_values = outputs.past_key_values
    outputs = model.generate(
        decoder_input_ids=decoder_input_ids,
        encoder_outputs=encoder_outputs,
        past_key_values=past_key_values,
        temperature=0.25,
        use_cache=True,
        **kwargs,
    )

    return outputs


def transcribe(batch):
    outputs = prompt_generation(batch["input_features"])
    transcriptions = processor.batch_decode(outputs, skip_special_tokens=True)
    batch["transcription"] = transcriptions

    return batch


half_sec_silence = np.zeros(8_000)


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        input_features = [
            {
                "input_features": processor.feature_extractor(
                    np.concatenate([prompt_wav, half_sec_silence, feature["audio"]]),
                    sampling_rate=16_000,
                ).input_features[0]
            }
            for feature in features
        ]
        batch = {
            "input_features": self.processor.feature_extractor.pad(
                input_features, return_tensors="pt"
            )
            .to(device)
            .to(torch_dtype),
            "audio_file": [feature["audio_file"] for feature in features],
        }

        return batch


def get_input_features(batch):
    output_batch = {
        "audio_file": [],
        "audio": [],
    }

    for audio in batch["audio"]:
        try:
            wav = librosa.load(audio, sr=16_000)[0]
            duration = wav.shape[0] // 16_000
            if (duration >= 5) or (
                duration <= 30
            ):  # below 30 seconds or above 5 seconds
                output_batch["audio_file"].append(audio)
                output_batch["audio"].append(wav)
        except Exception as e:
            print(f"Error processing {audio}: {e}")

    return output_batch


def main(root_folder: str, batch_size=64, num_proc=128):
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
    mp3_files = glob.glob(os.path.join(root_folder, "*.mp3")) + glob.glob(
        os.path.join(root_folder, "**/*.mp3"), recursive=True
    )

    os.makedirs("transcriptions", exist_ok=True)

    print("Total audio files:", len(mp3_files))

    ds = Dataset.from_dict({"audio": mp3_files})
    ds = ds.map(get_input_features, batched=True, num_proc=num_proc)
    dataloader = DataLoader(ds, batch_size=batch_size, collate_fn=data_collator)
    results = {"audio_file": [], "transcription": []}

    # resume from last checkpoint
    if os.path.exists("transcriptions/whisper_v3.csv"):
        df = pd.read_csv("transcriptions/whisper_v3.csv")
        results = {
            "audio_file": df["audio_file"].tolist(),
            "transcription": df["transcription"].tolist(),
        }
        ds = ds.filter(lambda x: x["audio_file"] not in results["audio_file"])
        dataloader = DataLoader(ds, batch_size=batch_size, collate_fn=data_collator)

    print("Transcribing...")

    for batch in tqdm(dataloader, total=len(ds) // batch_size):
        outputs = transcribe(batch)
        results["audio_file"].extend(batch["audio_file"])
        results["transcription"].extend(outputs["transcription"])

        # Save every 10,000 transcriptions
        if len(results["audio_file"]) % 10_000 == 0 and len(results["audio_file"]) > 0:
            df = pd.DataFrame(results)
            df.to_csv("transcriptions/whisper_v3.csv", index=False)

    df = pd.DataFrame(results)
    df = df[["audio_file", "transcription"]]
    df.to_csv("transcriptions/whisper_v3.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_root_path", required=True, type=str)
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--num_proc", default=64, type=int)
    args = parser.parse_args()

    main(args.audio_root_path, args.batch_size, args.num_proc)
