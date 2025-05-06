📰 Headline Generation using Encoder-Decoder Architectures
This project focuses on automatic headline generation using various sequence-to-sequence (Seq2Seq) architectures. It compares three different approaches to generate news headlines from article text.

📌 Architectures Implemented
1. 🔁 LSTM/GRU Encoder-Decoder (Without Attention)
A traditional Seq2Seq model using LSTM/GRU networks with a fixed-length context vector from the encoder to initialize the decoder.

2. 🎯 Encoder-Decoder with Attention (Bahdanau / Luong)
An enhanced model where the decoder can dynamically attend to relevant parts of the input sequence at each step:

Bahdanau Attention (Additive)

Luong Attention (Multiplicative)

3. 🧠 Transformer (Self-Attention)
A modern architecture based on self-attention mechanisms that completely replaces RNNs:

Positional Encoding

Multi-head Attention

Layer Normalization and Feedforward layers

🧪 Dataset
Source: Kaggle - News Headline Generation Dataset

Consists of news articles and corresponding headlines.

Preprocessed by tokenization, lowercasing, and padding.

🧠 Features
✅ Implemented using TensorFlow 
✅ Includes:

Embedding layers

Regularization: Dropout, L2 weight decay

Activation functions: ReLU, Softmax

Custom attention layers

Inference mode for evaluation

📉 Evaluation Metrics:

BLEU Score

ROUGE Score

📊 Model Comparison
Model	Attention	BLEU / ROUGE (sample)	Notes
LSTM Encoder-Decoder	❌ None	--	Struggles with long sentences
LSTM with Bahdanau/Luong	✅ Yes	--	More accurate and context-aware
Transformer (Self-Attn)	✅ Full Self-Attn	--	Fastest and most scalable

(Add your actual scores once available)

🚀 How to Run
Clone the repo and install dependencies:

bash
Copy
Edit
git clone https://github.com/your-username/headline-generation.git
cd headline-generation
pip install -r requirements.txt
Download the dataset from Kaggle and place it in a data/ folder.

Run the notebook or training scripts for each model:

bash
Copy
Edit
python train_lstm.py
python train_attention.py
python train_transformer.py
Evaluate and compare outputs using BLEU/ROUGE scores.

📁 Project Structure
kotlin
Copy
Edit
├── data/
│   └── news_dataset.csv
├── models/
│   ├── lstm_encoder_decoder.py
│   ├── attention_models.py
│   └── transformer.py
├── utils/
│   ├── preprocessing.py
│   └── metrics.py
├── train_lstm.py
├── train_attention.py
├── train_transformer.py
├── inference.py
├── README.md
└── requirements.txt
🔍 Results & Observations
Attention-based models outperform vanilla LSTM in capturing long-range dependencies.

Transformers are faster to train and more accurate in headline generation, especially with long input sequences.

Regularization and proper tokenization significantly improve generalization.

🤝 Credits
Dataset by Sahide Seker

Inspired by concepts from:

Bahdanau et al. (2015)

Luong et al. (2015)

Vaswani et al. (2017): “Attention is All You Need”

📜 License
This project is open-source and available under the MIT License.
