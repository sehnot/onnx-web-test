import json
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim

# **Zeichen-zu-Index-Mapping erstellen**
def create_vocab(data_file):
    with open(data_file, 'r') as f:
        data = json.load(f)
    # Sammle alle Zeichen aus Eingabe und Zielwerten
    all_text = "".join([
        d["input"] + d["output"]["name"] + d["output"]["company"] +
        d["output"]["website"] + d["output"]["phone"] for d in data
    ])
    unique_chars = sorted(set(all_text))  # Alle eindeutigen Zeichen
    char_to_index = {char: idx for idx, char in enumerate(unique_chars)}
    index_to_char = {idx: char for char, idx in char_to_index.items()}
    return char_to_index, index_to_char

# **Dataset-Klasse**
class SignatureDataset(Dataset):
    def __init__(self, data_file, char_to_index, max_input_len=256, max_output_len=128):
        with open(data_file, 'r') as f:
            data = json.load(f)
        self.inputs = [d['input'] for d in data]
        self.outputs = [d['output'] for d in data]
        self.char_to_index = char_to_index
        self.max_input_len = max_input_len
        self.max_output_len = max_output_len

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input_text = self.inputs[idx]
        output = self.outputs[idx]

        # Eingabesequenz in Zeichen-Indizes konvertieren
        input_vector = [self.char_to_index.get(c, 0) for c in input_text[:self.max_input_len]]
        input_vector += [0] * (self.max_input_len - len(input_vector))  # Padding

        # Zielsequenz erstellen und mit "|" trennen
        output_text = output["name"] + "|" + output["company"] + "|" + output["website"] + "|" + output["phone"]
        output_vector = [self.char_to_index.get(c, 0) for c in output_text[:self.max_output_len]]
        output_vector += [0] * (self.max_output_len - len(output_vector))  # Padding

        return torch.tensor(input_vector, dtype=torch.long), torch.tensor(output_vector, dtype=torch.long)

# **Seq2Seq-Modell**
class Seq2SeqModel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, max_output_len):
        super(Seq2SeqModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.encoder = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.decoder = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.max_output_len = max_output_len  # Als Attribut speichern

    def forward(self, x):
        # Eingabe einbetten
        x = self.embedding(x)
        # Encoder-Ausgabe
        encoder_output, (hidden, cell) = self.encoder(x)
        # Initiales Decoder-Input (leer, gepaddet)
        batch_size = x.size(0)
        decoder_input = torch.zeros(batch_size, self.max_output_len, self.embedding.embedding_dim, device=x.device)
        # Decoder-Ausgabe
        decoder_output, _ = self.decoder(decoder_input, (hidden, cell))
        # Zeichenklassifikation
        output = self.fc(decoder_output)
        return output

# **Trainingsparameter**
data_file = 'training_data.json'
char_to_index, index_to_char = create_vocab(data_file)
vocab_size = len(char_to_index)
embed_size = 64
hidden_size = 128
max_input_len = 256
max_output_len = 128

# **Dataset und DataLoader**
dataset = SignatureDataset(data_file, char_to_index, max_input_len=max_input_len, max_output_len=max_output_len)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# **Modell, Optimierer und Verlustfunktion**
model = Seq2SeqModel(vocab_size, embed_size, hidden_size, max_output_len)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# **Training**
epochs = 50
for epoch in range(epochs):
    for inputs, outputs in dataloader:
        optimizer.zero_grad()
        # Vorhersage
        predictions = model(inputs)
        # Reshape für CrossEntropyLoss
        predictions = predictions.view(-1, vocab_size)
        outputs = outputs.view(-1)
        # Verlust berechnen und Backpropagation
        loss = criterion(predictions, outputs)
        loss.backward()
        optimizer.step()
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")

# Modell exportieren mit Batch-Größe 1
example_input = torch.randint(0, vocab_size, (1, max_input_len))  # Batch-Size = 1
torch.onnx.export(
    model,
    example_input,
    "signature_extractor_seq2seq.onnx",
    input_names=["input"],
    output_names=["output"]
)
print("Model successfully saved with batch size 1!")
