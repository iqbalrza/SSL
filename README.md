# 🔐 SSL TMIX — Semi-Supervised Learning

Implementasi **Semi-Supervised Learning (SSL)** menggunakan metode TMIX untuk klasifikasi teks dengan memanfaatkan data berlabel sedikit dan data tak berlabel dalam jumlah besar.

## 📌 Deskripsi

SSL TMIX menggabungkan teknik interpolasi fitur (Mixup) pada ruang embedding untuk meningkatkan performa model NLP ketika data berlabel sangat terbatas.

## 🛠️ Teknologi

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)

**Library yang digunakan:**
- `transformers` (HuggingFace) — Pre-trained language model
- `torch` (PyTorch) — Framework deep learning
- `scikit-learn` — Evaluasi dan preprocessing
- `pandas`, `numpy` — Manipulasi data

## 📁 Struktur File

```
└── SSL TMIX.py    # Script utama implementasi SSL TMIX
```

## 📊 Dataset & Model

- **Task:** Klasifikasi teks semi-supervised
- **Metode:** TMIX (Text Mixup untuk SSL)
- **Model dasar:** Pre-trained Transformer (BERT/RoBERTa)
- **Setting:** Labeled data kecil + Unlabeled data besar

## 🚀 Cara Menjalankan

1. Clone repo ini:
   ```bash
   git clone https://github.com/iqbalreza/SSL.git
   ```
2. Install dependencies:
   ```bash
   pip install torch transformers scikit-learn pandas numpy
   ```
3. Jalankan script:
   ```bash
   python "SSL TMIX.py"
   ```

## 👤 Author

**Iqbal Reza** — Universitas Komputer Indonesia (UNIKOM)

## 📄 Lisensi

Proyek ini dibuat untuk keperluan penelitian akademik.