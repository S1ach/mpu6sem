import time
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from collections import Counter
from torch.utils.data import Dataset, DataLoader, TensorDataset
from tqdm import tqdm

data = pd.read_csv("reviews_preprocessed.csv")

all_words = " ".join(data.processed.values).split()
counter = Counter(all_words)

vocabulary = sorted(counter, key=counter.get, reverse=True)

int2word = dict(enumerate(vocabulary, 1))
int2word[0] = "PAD"

word2int = {word: id for id, word in int2word.items()}

reviews = data.processed.values
all_words = " ".join(reviews).split()

review_enc = [[word2int[word] for word in review.split()] for review in reviews]

sequence_length = 256
reviews_padding = np.full((len(review_enc), sequence_length), word2int['PAD'], dtype=int)

for i, row in enumerate(review_enc):
    reviews_padding[i, :len(row)] = np.array(row)[:sequence_length]

labels = data.label.to_numpy()

perm = np.random.permutation(len(reviews_padding))
reviews_padding, labels = reviews_padding[perm], labels[perm]

train_len = 0.6
test_len = 0.5
train_last_index = int(len(reviews_padding) * train_len)

train_x, remainder_x = reviews_padding[:train_last_index], reviews_padding[train_last_index:]
train_y, remainder_y = labels[:train_last_index], labels[train_last_index:]

test_last_index = int(len(remainder_x) * test_len)
test_x = remainder_x[:test_last_index]
test_y = remainder_y[:test_last_index]

check_x = remainder_x[test_last_index:]
check_y = remainder_y[test_last_index:]

train_dataset = TensorDataset(torch.from_numpy(train_x.copy()), torch.from_numpy(train_y.copy()))
test_dataset = TensorDataset(torch.from_numpy(test_x.copy()), torch.from_numpy(test_y.copy()))
check_dataset = TensorDataset(torch.from_numpy(check_x.copy()), torch.from_numpy(check_y.copy()))

batch_size = 128
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
test_loader = DataLoader(test_dataset, shuffle=True, batch_size=batch_size)
check_loader = DataLoader(check_dataset, shuffle=True, batch_size=batch_size)


class TextModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, dropout=0.3):
        super(TextModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, text, text_lengths):
        embedded = self.embedding(text)
        packed_embedded = nn.utils.rnn.pack_padded_sequence(
            embedded, text_lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_output, (hidden, cell) = self.lstm(packed_embedded)
        hidden_last = hidden[-1, :, :]
        dropped = self.dropout(hidden_last)
        output = self.fc(dropped)
        predictions = self.sigmoid(output)
        return predictions


def create_model(vocab_size, embedding_dim=100, hidden_dim=256, output_dim=1, n_layers=4, dropout=0.3):
    model = TextModel(vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, dropout)
    return model


vocab_size = len(word2int)
model = create_model(vocab_size, n_layers=4)

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters())


def accuracy(predictions, labels):
    rounded_preds = torch.round(predictions)
    correct = (rounded_preds == labels).float()
    return correct.sum() / len(correct)


model_path = './lern_4layers.pth'
best_loss = float('inf')
num_epochs = 5

total_start = time.time()

for epoch in range(num_epochs):
    epoch_start = time.time()

    model.train()
    train_loss = 0
    train_acc = 0

    train_bar = tqdm(train_loader, desc=f'Эпоха {epoch + 1}/{num_epochs} [train]', leave=False)
    for batch_x, batch_y in train_bar:
        batch_x = batch_x.long()
        batch_y = batch_y.float().unsqueeze(1)

        lengths = (batch_x != word2int['PAD']).sum(dim=1).clamp(min=1)

        optimizer.zero_grad()
        predictions = model(batch_x, lengths)
        loss = criterion(predictions, batch_y)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
        optimizer.step()

        acc = accuracy(predictions, batch_y)
        train_loss += loss.item()
        train_acc += acc.item()

        train_bar.set_postfix(loss=f'{loss.item():.4f}', acc=f'{acc.item():.4f}')

    avg_train_loss = train_loss / len(train_loader)
    avg_train_acc = train_acc / len(train_loader)

    model.eval()
    test_loss = 0
    test_acc = 0

    with torch.no_grad():
        test_bar = tqdm(test_loader, desc=f'Эпоха {epoch + 1}/{num_epochs} [test] ', leave=False)
        for batch_x, batch_y in test_bar:
            batch_x = batch_x.long()
            batch_y = batch_y.float().unsqueeze(1)

            lengths = (batch_x != word2int['PAD']).sum(dim=1).clamp(min=1)

            predictions = model(batch_x, lengths)
            loss = criterion(predictions, batch_y)
            acc = accuracy(predictions, batch_y)

            test_loss += loss.item()
            test_acc += acc.item()

            test_bar.set_postfix(loss=f'{loss.item():.4f}', acc=f'{acc.item():.4f}')

    avg_test_loss = test_loss / len(test_loader)
    avg_test_acc = test_acc / len(test_loader)

    epoch_time = time.time() - epoch_start

    if avg_test_loss < best_loss:
        best_loss = avg_test_loss
        torch.save(model.state_dict(), model_path)
        print(f'Эпоха {epoch + 1}. Сохранена лучшая модель (loss: {avg_test_loss:.4f})')

    print(
        f'Эпоха {epoch + 1}: '
        f'Train Accuracy = {avg_train_acc:.4f}, '
        f'Test Accuracy = {avg_test_acc:.4f}, '
        f'время эпохи = {epoch_time:.1f} сек'
    )

total_time = time.time() - total_start
print(f'\nОбщее время обучения ({num_epochs} эпох, n_layers=4): {total_time:.1f} сек')