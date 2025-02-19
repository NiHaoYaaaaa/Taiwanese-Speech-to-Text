import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
from torchaudio.transforms import MelSpectrogram, Resample
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from torch.nn.utils.rnn import pad_sequence

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Hyperparameters
batch_size = 16
epochs = 10
learning_rate = 0.001

# Dataset processing class
class SpeechDataset(Dataset):
    def __init__(self, csv_file, data_folder, transform=None):
        self.annotations = pd.read_csv(csv_file, dtype={0: str})  # Ensure the first column is of string type
        self.data_folder = data_folder
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        audio_path = os.path.join(self.data_folder, self.annotations.iloc[index, 0] + ".wav")
        waveform, sample_rate = torchaudio.load(audio_path)

        if self.transform:
            waveform = self.transform(waveform)

        label = self.annotations.iloc[index, 1]
        return waveform, label

# Audio transformation
transform = nn.Sequential(
    Resample(orig_freq=44100, new_freq=16000),
    MelSpectrogram(sample_rate=16000, n_fft=400, hop_length=160, n_mels=64)  # Adjust parameters to avoid zero filters
)

# Custom collate_fn for padding audio sequences
def collate_fn(batch):
    waveforms = [item[0].squeeze(0).T for item in batch]  # Extract waveforms and adjust dimensions to (seq_len, features)
    labels = [item[1] for item in batch]  # Extract corresponding text labels

    # Pad waveforms to ensure uniform seq_len within the batch
    waveforms_padded = pad_sequence(waveforms, batch_first=True).permute(0, 2, 1)  # Adjust back to (batch, features, seq_len)

    return waveforms_padded, labels

# Load training data
train_dataset = SpeechDataset(csv_file="./train-toneless.csv", data_folder="./train", transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

# RNN model
class SpeechRecognitionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2):
        super(SpeechRecognitionModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.fc(x)
        return x

# Initialize model
input_dim = 64  # Number of Mel spectrogram features
hidden_dim = 256
output_dim = len(set("abcdefghijklmnopqrstuvwxyz ")) + 1  # Including blank character
model = SpeechRecognitionModel(input_dim, hidden_dim, output_dim).to(device)

# CTC loss and optimizer
criterion = nn.CTCLoss(blank=output_dim - 1)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    for batch in train_loader:
        waveform, label = batch
        waveform = waveform.to(device)

        # Calculate actual sequence lengths for inputs
        input_lengths = torch.tensor([wave.shape[-1] for wave in waveform], dtype=torch.long).to(device)

        # Assume feature lengths are fixed
        inputs = waveform.permute(0, 2, 1)  # Adjust dimensions to match LSTM (batch, seq, feature)

        # Flatten labels into a single tensor and calculate target lengths
        flattened_targets = []
        target_lengths = []

        for l in label:
            target = [ord(c) - ord('a') for c in l]
            flattened_targets.extend(target)
            target_lengths.append(len(target))

        flattened_targets = torch.tensor(flattened_targets, dtype=torch.long).to(device)
        target_lengths = torch.tensor(target_lengths, dtype=torch.long).to(device)

        optimizer.zero_grad()
        outputs = model(inputs)

        # Ensure input_lengths matches the batch size
        loss = criterion(outputs.log_softmax(2), flattened_targets, input_lengths, target_lengths)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}")

# Transcription function
def transcribe(audio_path, model):
    model.eval()
    waveform, _ = torchaudio.load(audio_path)
    waveform = transform(waveform).unsqueeze(0).to(device)
    outputs = model(waveform.permute(0, 2, 1))
    predicted = torch.argmax(outputs, dim=2).squeeze(0).tolist()
    transcription = "".join([chr(p + ord('a')) if p < 26 else " " for p in predicted])
    return transcription.strip()

# Process test data and generate submission.csv
test_folder = "./test"
output_csv = "submission.csv"
results = []

for file_name in os.listdir(test_folder):
    if file_name.endswith(".wav"):
        audio_path = os.path.join(test_folder, file_name)
        print(f"Processing {file_name}...")
        transcription = transcribe(audio_path, model)
        results.append({"id": file_name.split(".")[0], "text": transcription})

# Save results to CSV
submission_df = pd.DataFrame(results)
submission_df.to_csv(output_csv, index=False)
print(f"Transcriptions saved to {output_csv}")
