import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Embedding,
    LSTM,
    Dense,
    Dropout,
    Bidirectional,
    LayerNormalization,
    Attention,
    MultiHeadAttention,
    Input,
    Concatenate,
    GlobalMaxPooling1D,
)
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.regularizers import l2
import re
import pickle
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


class NextWordPredictor:
    def __init__(
        self,
        vocab_size=10000,
        embedding_dim=300,
        lstm_units=256,
        max_seq_length=50,
        dropout_rate=0.3,
    ):
        """
        Next Word Prediction Model with modern NLP techniques

        Args:
            vocab_size: Maximum vocabulary size
            embedding_dim: Embedding layer dimension
            lstm_units: LSTM hidden units
            max_seq_length: Maximum sequence length
            dropout_rate: Dropout rate for regularization
        """
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.lstm_units = lstm_units
        self.max_seq_length = max_seq_length
        self.dropout_rate = dropout_rate
        self.tokenizer = None
        self.model = None
        self.history = None

    def preprocess_text(self, text):
        """text preprocessing"""
        # Convert to lowercase
        text = text.lower()

        # Remove extra whitespace and normalize
        text = re.sub(r"\s+", " ", text)

        # Keep important punctuation, remove others
        text = re.sub(r"[^\w\s.,!?;:-]", "", text)

        # Split into sentences for better context
        sentences = re.split(r"[.!?]+", text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]

        return sentences

    def create_sequences(self, sentences):
        """Create training sequences with sliding window approach"""
        sequences = []

        for sentence in sentences:
            tokens = self.tokenizer.texts_to_sequences([sentence])[0]

            # Create overlapping sequences
            for i in range(1, len(tokens)):
                # Variable length sequences for better learning
                start_idx = max(0, i - self.max_seq_length + 1)
                seq = tokens[start_idx : i + 1]
                if len(seq) >= 2:  # Minimum sequence length
                    sequences.append(seq)

        return sequences

    def prepare_data(self, text_file_path):
        """Prepare training data with preprocessing"""
        try:
            with open(text_file_path, "r", encoding="utf-8") as f:
                text = f.read()
        except FileNotFoundError:
            # Fallback: create sample text for demonstration
            text = """
            Sherlock Holmes was a consulting detective. He lived at 221B Baker Street.
            Dr. Watson was his loyal companion and friend. Together they solved many mysteries.
            Holmes had extraordinary deductive abilities. He could solve the most complex cases.
            The detective used logic and observation. His methods were revolutionary for the time.
            """
            print("Warning: Using sample text. Please provide the actual text file.")

        # Preprocess text
        sentences = self.preprocess_text(text)

        # Initialize tokenizer with advanced options
        self.tokenizer = Tokenizer(
            num_words=self.vocab_size,
            oov_token="<OOV>",
            filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
        )

        # Fit tokenizer on sentences
        self.tokenizer.fit_on_texts(sentences)

        # Update vocab size to actual size
        self.vocab_size = len(self.tokenizer.word_index) + 1

        # Create sequences
        sequences = self.create_sequences(sentences)

        if not sequences:
            raise ValueError("No valid sequences created. Check your input text.")

        # Pad sequences
        sequences = pad_sequences(
            sequences, maxlen=self.max_seq_length, padding="pre", truncating="pre"
        )

        # Split into X and y
        X = sequences[:, :-1]
        y = sequences[:, -1]

        # Convert y to categorical
        y = tf.keras.utils.to_categorical(y, num_classes=self.vocab_size)

        return X, y

    def build_model(self):
        """Build model with modern architecture"""
        # Input layer
        input_layer = Input(shape=(self.max_seq_length - 1,))

        # Embedding layer with regularization
        embedding = Embedding(
            input_dim=self.vocab_size,
            output_dim=self.embedding_dim,
            mask_zero=True,
            embeddings_regularizer=l2(0.001),
        )(input_layer)

        # Bidirectional LSTM layers with residual connections
        lstm1 = Bidirectional(
            LSTM(self.lstm_units, return_sequences=True, dropout=self.dropout_rate)
        )(embedding)
        lstm1_norm = LayerNormalization()(lstm1)

        lstm2 = Bidirectional(
            LSTM(self.lstm_units // 2, return_sequences=True, dropout=self.dropout_rate)
        )(lstm1_norm)
        lstm2_norm = LayerNormalization()(lstm2)

        # Self-attention mechanism
        attention = MultiHeadAttention(num_heads=8, key_dim=self.lstm_units // 4)(
            lstm2_norm, lstm2_norm
        )

        # Combine LSTM and attention outputs
        combined = Concatenate()([lstm2_norm, attention])

        # Global pooling to get fixed-size representation
        pooled = GlobalMaxPooling1D()(combined)

        # Dense layers with residual connection
        dense1 = Dense(512, activation="relu", kernel_regularizer=l2(0.001))(pooled)
        dense1_dropout = Dropout(self.dropout_rate)(dense1)
        dense1_norm = LayerNormalization()(dense1_dropout)

        dense2 = Dense(256, activation="relu", kernel_regularizer=l2(0.001))(
            dense1_norm
        )
        dense2_dropout = Dropout(self.dropout_rate)(dense2)

        # Output layer
        output = Dense(self.vocab_size, activation="softmax")(dense2_dropout)

        # Create model
        self.model = Model(inputs=input_layer, outputs=output)

        # Use AdamW optimizer with learning rate scheduling
        optimizer = AdamW(learning_rate=0.001, weight_decay=0.01)

        self.model.compile(
            loss="categorical_crossentropy",
            optimizer=optimizer,
            metrics=["accuracy", "top_k_categorical_accuracy"],
        )

        return self.model

    def train(self, X, y, validation_split=0.2, epochs=50, batch_size=64):
        """Train model with callbacks and monitoring"""
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=42
        )

        # Define callbacks
        callbacks = [
            EarlyStopping(
                monitor="val_loss", patience=10, restore_best_weights=True, verbose=1
            ),
            ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=5, min_lr=1e-7, verbose=1
            ),
            ModelCheckpoint(
                "best_model.h5", monitor="val_loss", save_best_only=True, verbose=1
            ),
        ]

        # Train model
        self.history = self.model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1,
        )

        return self.history

    def predict_next_words(self, seed_text, n_words=5, temperature=0.8, top_k=10):
        """
        Predict next words with temperature sampling and top-k filtering

        Args:
            seed_text: Input text
            n_words: Number of words to predict
            temperature: Sampling temperature (higher = more creative)
            top_k: Consider only top-k most likely words
        """
        if not self.model or not self.tokenizer:
            raise ValueError("Model not trained. Please train the model first.")

        result = seed_text.lower()

        for _ in range(n_words):
            # Tokenize current text
            token_list = self.tokenizer.texts_to_sequences([result])[0]

            # Pad sequence
            token_list = pad_sequences(
                [token_list], maxlen=self.max_seq_length - 1, padding="pre"
            )

            # Get predictions
            predictions = self.model.predict(token_list, verbose=0)[0]

            # Apply temperature scaling
            predictions = np.log(predictions + 1e-10) / temperature
            predictions = np.exp(predictions)
            predictions = predictions / np.sum(predictions)

            # Get top-k predictions
            top_k_indices = np.argsort(predictions)[-top_k:]
            top_k_probs = predictions[top_k_indices]
            top_k_probs = top_k_probs / np.sum(top_k_probs)

            # Sample from top-k
            predicted_id = np.random.choice(top_k_indices, p=top_k_probs)

            # Find corresponding word
            output_word = ""
            for word, index in self.tokenizer.word_index.items():
                if index == predicted_id:
                    output_word = word
                    break

            if output_word:
                result += " " + output_word
            else:
                break

        return result

    def get_word_probabilities(self, seed_text, top_n=10):
        """Get probability distribution for next word"""
        if not self.model or not self.tokenizer:
            raise ValueError("Model not trained.")

        token_list = self.tokenizer.texts_to_sequences([seed_text.lower()])[0]
        token_list = pad_sequences(
            [token_list], maxlen=self.max_seq_length - 1, padding="pre"
        )

        predictions = self.model.predict(token_list, verbose=0)[0]

        # Get top predictions
        top_indices = np.argsort(predictions)[-top_n:][::-1]

        results = []
        for idx in top_indices:
            for word, word_idx in self.tokenizer.word_index.items():
                if word_idx == idx:
                    results.append((word, predictions[idx]))
                    break

        return results

    def save_model(self, filepath):
        """Save model and tokenizer"""
        self.model.save(f"{filepath}_model.keras")
        with open(f"{filepath}_tokenizer.pickle", "wb") as f:
            pickle.dump(self.tokenizer, f)

    def load_model(self, filepath):
        """Load model and tokenizer"""
        self.model = tf.keras.models.load_model(f"{filepath}_model.h5")
        with open(f"{filepath}_tokenizer.pickle", "rb") as f:
            self.tokenizer = pickle.load(f)

    def plot_training_history(self):
        """Plot training history"""
        if not self.history:
            print("No training history available.")
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        # Plot loss
        ax1.plot(self.history.history["loss"], label="Training Loss")
        ax1.plot(self.history.history["val_loss"], label="Validation Loss")
        ax1.set_title("Model Loss")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.legend()

        # Plot accuracy
        ax2.plot(self.history.history["accuracy"], label="Training Accuracy")
        ax2.plot(self.history.history["val_accuracy"], label="Validation Accuracy")
        ax2.set_title("Model Accuracy")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy")
        ax2.legend()

        plt.tight_layout()
        plt.show()


# Example usage
if __name__ == "__main__":
    # Initialize predictor
    predictor = NextWordPredictor(
        vocab_size=10000,
        embedding_dim=300,
        lstm_units=256,
        max_seq_length=50,
        dropout_rate=0.3,
    )

    # Prepare data (replace with your actual file path)
    try:
        X, y = predictor.prepare_data(
            "/mnt/data/sherlock-holm.es_stories_plain-text_advs.txt"
        )
        print(
            f"Data prepared: {X.shape[0]} sequences, vocab size: {predictor.vocab_size}"
        )

        # Build model
        model = predictor.build_model()
        print("Model built successfully!")
        print(model.summary())

        # Train model
        print("Starting training...")
        history = predictor.train(X, y, epochs=30)

        # Make predictions
        print("\n--- Predictions ---")
        seed_text = "to sherlock holmes"

        # Standard prediction
        result = predictor.predict_next_words(seed_text, n_words=5, temperature=0.7)
        print(f"Input: {seed_text}")
        print(f"Output: {result}")

        # Creative prediction (higher temperature)
        creative_result = predictor.predict_next_words(
            seed_text, n_words=5, temperature=1.2
        )
        print(f"Creative: {creative_result}")

        # Conservative prediction (lower temperature)
        conservative_result = predictor.predict_next_words(
            seed_text, n_words=5, temperature=0.3
        )
        print(f"Conservative: {conservative_result}")

        # Show word probabilities
        print("\n--- Word Probabilities ---")
        probs = predictor.get_word_probabilities(seed_text, top_n=10)
        for word, prob in probs:
            print(f"{word}: {prob:.4f}")

        # Plot training history
        predictor.plot_training_history()

        # Save model
        predictor.save_model("nlp_model")
        print("Model saved successfully!")

    except Exception as e:
        print(f"Error: {e}")
        print("Running with sample data for demonstration...")

        # Create sample data for demo
        sample_text = """
        Sherlock Holmes was a brilliant detective who lived in London. He solved many mysterious cases with his friend Dr. Watson.
        The detective had extraordinary powers of observation and deduction. His methods were scientific and logical.
        Holmes could solve the most complex mysteries by noticing small details that others missed.
        Dr. Watson documented their adventures and published them as stories.
        The consulting detective helped Scotland Yard solve difficult cases.
        """

        # Write sample text to temporary file
        with open("sample_text.txt", "w") as f:
            f.write(sample_text * 20)  # Repeat for more data

        X, y = predictor.prepare_data("sample_text.txt")
        model = predictor.build_model()
        history = predictor.train(X, y, epochs=20)

        # Make predictions
        result = predictor.predict_next_words("sherlock holmes", n_words=3)
        print(f"Prediction: {result}")
