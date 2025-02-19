from datasets import Dataset, Audio
from transformers import (
    WhisperProcessor, 
    WhisperForConditionalGeneration, 
    Seq2SeqTrainingArguments, 
    Seq2SeqTrainer
) 
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import os
import torch
import librosa
import evaluate
import pandas as pd

model_id = 'openai/whisper-small'
out_dir = 'whisper_finetune_output'
epochs = 10
# batch 16->32, back to 16
batch_size = 16

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available
print(device)

# Load custom dataset from CSV
csv_file = "./train-toneless.csv"  # Path to the CSV file
data_folder = "./train"  # Folder containing audio files
df = pd.read_csv(csv_file)
df["audio"] = df["id"].apply(lambda x: f"{data_folder}/{x}.wav")

# Convert DataFrame to HuggingFace Dataset
dataset = Dataset.from_pandas(df)
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

# Split dataset into train and validation sets
dataset_split = dataset.train_test_split(test_size=0.05, seed=42)
atc_dataset_train = dataset_split["train"]
atc_dataset_valid = dataset_split["test"]

print(atc_dataset_train)
print(atc_dataset_valid)

# feature_extractor = WhisperFeatureExtractor.from_pretrained(model_id)
# processing_class = WhisperTokenizer.from_pretrained(model_id, language='Chinese', task='transcribe')
# Initialize the processor
processor = WhisperProcessor.from_pretrained(model_id, language="Chinese", task="transcribe")

# Load the model and move it to GPU
model = WhisperForConditionalGeneration.from_pretrained(model_id).to(device)

def prepare_dataset(batch):
    audio = batch['audio']
    batch['input_features'] = processor.feature_extractor(audio['array'], sampling_rate=audio['sampling_rate']).input_features[0]
    batch['labels'] = processor.tokenizer(batch['text']).input_ids
    return batch

# Preprocess the train dataset
atc_dataset_train = atc_dataset_train.map(
    prepare_dataset, 
    num_proc=1
)

# Preprocess the validation dataset
atc_dataset_valid = atc_dataset_valid.map(
    prepare_dataset, 
    num_proc=1
)

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{'input_features': feature['input_features']} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors='pt')

        label_features = [{'input_ids': feature['labels']} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors='pt')

        labels = labels_batch['input_ids'].masked_fill(labels_batch.attention_mask.ne(1), -100)

        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch['labels'] = labels

        return batch

# Data collator for padding inputs and labels
data_collator = DataCollatorSpeechSeq2SeqWithPadding(
    processor=processor,
    decoder_start_token_id=model.config.decoder_start_token_id,
)

# Load WER metric for evaluation
metric = evaluate.load('wer')

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # Replace -100 with the pad_token_id
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    # Decode predictions and references
    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.batch_decode(label_ids, skip_special_tokens=True)

    # Compute WER (Word Error Rate)
    wer = 100 * metric.compute(predictions=pred_str, references=label_str)

    return {'wer': wer}

# Training arguments for the Seq2Seq Trainer
training_args = Seq2SeqTrainingArguments(
    output_dir=out_dir, 
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    gradient_accumulation_steps=1, 
    learning_rate=2e-5,
    warmup_ratio=0.4,
    bf16=True if torch.cuda.is_bf16_supported() else False,  # Enable bf16 if supported
    fp16=False if torch.cuda.is_bf16_supported() else True,  # Disable fp16 if bf16 is enabled
    num_train_epochs=epochs,
    eval_strategy='epoch',
    logging_strategy='epoch',
    save_strategy='epoch',
    predict_with_generate=True,
    generation_max_length=225,
    report_to=['tensorboard'],
    load_best_model_at_end=True,
    metric_for_best_model='wer',
    greater_is_better=False,
    dataloader_num_workers=0,
    save_total_limit=2,
    lr_scheduler_type='linear',
    seed=42,
    data_seed=42
)

# Initialize Seq2Seq Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=atc_dataset_train,
    eval_dataset=atc_dataset_valid,
    processing_class=processor,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# Train the model
trainer.train()

# Save the best model and processor
model.save_pretrained(f"{out_dir}/best_model")
processor.save_pretrained(f"{out_dir}/best_model")

# Inference on test data folder
def transcribe_audio(audio_path):
    audio, _ = librosa.load(audio_path, sr=16000)
    input_features = processor(audio, sampling_rate=16000, return_tensors="pt").input_features.to(device)
    predicted_ids = model.generate(input_features)
    transcription = processor.decode(predicted_ids[0], skip_special_tokens=True)
    return transcription

test_folder = "./test"  # Folder containing audio files to process
output_csv = "submission.csv"
results = []

for file_name in os.listdir(test_folder):
    if file_name.endswith(".wav"):
        audio_path = os.path.join(test_folder, file_name)
        # print(f"Processing {file_name}...")
        transcription = transcribe_audio(audio_path)
        results.append({"id": file_name.split(".")[0], "text": transcription})

# Save results to submission.csv
submission_df = pd.DataFrame(results)
submission_df.to_csv(output_csv, index=False)
print(f"Transcriptions saved to {output_csv}")

# scheduler='linear', warmup_ratio=0.2, batch=32
# train:0.9 & valid:0.1, warmup_ratio=0.3
# batch=16
# train:0.95 & valid:0.05, lr_rate=2e-5, warmup_ratio=0.4
# next try: 