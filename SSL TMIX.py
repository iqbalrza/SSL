import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np

# 1. Dataset Custom dengan Preprocessing TF-IDF
class CustomDataset(Dataset):
    def __init__(self, data, vectorizer, is_unlabeled=False):
        self.data = data
        self.vectorizer = vectorizer
        self.features = self.vectorizer.transform(data['entry']).toarray()
        if not is_unlabeled:
            self.labels = data['label'].apply(lambda x: 1 if x == "positive" else 0).values
        else:
            self.labels = [-1] * len(data)  # Placeholder untuk data tidak berlabel

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        feature = torch.tensor(self.features[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return feature, label

# 2. Definisi Model Classifier
class SimpleClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(SimpleClassifier, self).__init__()
        self.fc = nn.Linear(input_size, num_classes)

    def forward(self, x):
        return self.fc(x)

# 3. Weighted Average
def weighted_average(outputs):
    weights = torch.tensor([1.0 / len(outputs)] * len(outputs))  # Bobot rata-rata
    return sum(w * o for w, o in zip(weights, outputs))

# 4. Sharpening Function
def sharpen(p, T=0.5):
    p = p ** (1 / T)
    return p / p.sum(dim=-1, keepdim=True)

# 5. Entropy Minimization Loss
def entropy_loss_fn(p):
    p = p + 1e-8  # Avoid log(0)
    entropy = -torch.sum(p * torch.log(p), dim=-1)
    return torch.mean(entropy)

# 6. TMix (MixUp) Function for Augmentation
def mixup_data(x, y, alpha=0.2):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size()[0]
    index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

# 7. Loss Function with MixUp
def mixup_loss_fn(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# 8. Muat Data Berlabel dan Tidak Berlabel
labeled_data = pd.read_csv(r"C:\Users\iqbal\OneDrive\Documents\Code Labs\ssl\labeled_data.csv")
unlabeled_data = pd.read_csv(r"C:\Users\iqbal\OneDrive\Documents\Code Labs\ssl\unlabeled_data.csv")

# 9. Inisialisasi Vectorizer dan Dataset
vectorizer = TfidfVectorizer(max_features=100)
vectorizer.fit(labeled_data['entry'])  # Fit hanya dengan data berlabel

labeled_dataset = CustomDataset(labeled_data, vectorizer)
unlabeled_dataset = CustomDataset(unlabeled_data, vectorizer, is_unlabeled=True)

train_loader = DataLoader(labeled_dataset, batch_size=32, shuffle=True)
unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=32)

# 10. Inisialisasi Model dan Optimizer
input_size = 100  # Fitur TF-IDF
num_classes = 2  # Positive dan Negative
models = [SimpleClassifier(input_size, num_classes) for _ in range(3)]
optimizers = [optim.Adam(model.parameters(), lr=0.001) for model in models]

# 11. Training Loop dengan Data Berlabel dan Tidak Berlabel
for epoch in range(10):
    # --- Training dengan data berlabel ---
    for inputs, targets in train_loader:
        # Augmentasi dengan MixUp
        inputs, targets_a, targets_b, lam = mixup_data(inputs, targets)
        
        # Training pada data berlabel
        outputs = [model(inputs) for model in models]
        final_output = weighted_average(outputs)
        
        loss = mixup_loss_fn(nn.CrossEntropyLoss(), final_output, targets_a, targets_b, lam)

        for optimizer in optimizers:
            optimizer.zero_grad()
        loss.backward()
        for optimizer in optimizers:
            optimizer.step()

    # --- Konsistensi Loss untuk data tidak berlabel ---
    for inputs, _ in unlabeled_loader:
        with torch.no_grad():
            # Prediksi pseudo-labels
            outputs = [model(inputs) for model in models]
            avg_output = weighted_average(outputs)
            sharpened_output = sharpen(avg_output)

            # Entropy minimization loss
            entropy_loss = entropy_loss_fn(sharpened_output)

    print(f"Epoch [{epoch + 1}/10], Loss: {loss.item():.4f}, Entropy Loss: {entropy_loss.item():.4f}")

# 12. Prediksi untuk Data Tidak Berlabel
def predict_unlabeled(models, loader):
    predictions = []

    with torch.no_grad():
        for inputs, _ in loader:
            outputs = [model(inputs) for model in models]
            final_output = weighted_average(outputs)
            _, predicted = torch.max(final_output, 1)
            predictions.extend(predicted.tolist())

    return predictions

# Lakukan prediksi pada data tidak berlabel
unlabeled_predictions = predict_unlabeled(models, unlabeled_loader)

# 13. Gabungkan Data dan Simpan sebagai CSV
unlabeled_data['label'] = ['positive' if p == 1 else 'negative' for p in unlabeled_predictions]
combined_data = pd.concat([labeled_data, unlabeled_data], ignore_index=True)
combined_data.to_csv(r"C:\Users\iqbal\OneDrive\Documents\Code Labs\ssl\prediction_results.csv", index=False)

combined_data.head()

# Misalkan 'predicted_data.csv' adalah hasil prediksi dari model
predicted_data = pd.read_csv(r"C:\Users\iqbal\OneDrive\Documents\Code Labs\ssl\prediction_results.csv")  # Ganti dengan path file prediksi yang tepat
actual_data = pd.read_csv(r'C:\Users\iqbal\OneDrive\Documents\Code Labs\ssl\labeled_text_combined_expanded.csv', delimiter=';')  # Data actual dari user

# Gabungkan kedua dataframe berdasarkan kolom 'entry' agar bisa dibandingkan
comparison = pd.merge(predicted_data, actual_data, on='entry', suffixes=('_predicted', '_actual'))

# Kolom baru untuk melihat apakah prediksi sesuai dengan label aktual
comparison['match'] = comparison['label_predicted'] == comparison['label_actual']

# Menghitung persentase kecocokan prediksi
accuracy = comparison['match'].mean() * 100

print("Hasil Perbandingan Prediksi:")
print(comparison.head())  # Menampilkan beberapa hasil perbandingan

print(f"Akurasi Prediksi: {accuracy:.2f}%")


