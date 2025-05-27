import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Embedding,
    LSTM,
    Dense,
    Dropout,
    Bidirectional,
    LayerNormalization,
)
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.callbacks import EarlyStopping
import re
import pickle
import io
import time
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

# Configure Streamlit page
st.set_page_config(
    page_title="AI Text Predictor",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for better styling
st.markdown(
    """
<style>
.main-header {
    font-size: 3rem;
    font-weight: bold;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
}
.prediction-box {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 10px;
    border-left: 5px solid #1f77b4;
    margin: 1rem 0;
}
.metric-container {
    background-color: #ffffff;
    padding: 1rem;
    border-radius: 10px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}
</style>
""",
    unsafe_allow_html=True,
)


class StreamlitNLPPredictor:
    def __init__(
        self,
        vocab_size=5000,
        embedding_dim=128,
        lstm_units=128,
        max_seq_length=30,
        dropout_rate=0.3,
    ):
        """Streamlit-optimized NLP predictor with smaller architecture for faster training"""
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.lstm_units = lstm_units
        self.max_seq_length = max_seq_length
        self.dropout_rate = dropout_rate
        self.tokenizer = None
        self.model = None

    def preprocess_text(self, text):
        """Clean and preprocess text"""
        text = text.lower()
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"[^\w\s.,!?;:-]", "", text)
        sentences = re.split(r"[.!?]+", text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 5]
        return sentences

    def create_sequences(self, sentences):
        """Create training sequences"""
        sequences = []
        for sentence in sentences:
            tokens = self.tokenizer.texts_to_sequences([sentence])[0]
            for i in range(1, len(tokens)):
                start_idx = max(0, i - self.max_seq_length + 1)
                seq = tokens[start_idx : i + 1]
                if len(seq) >= 2:
                    sequences.append(seq)
        return sequences

    def prepare_data(self, text):
        """Prepare training data from text"""
        sentences = self.preprocess_text(text)

        self.tokenizer = Tokenizer(
            num_words=self.vocab_size,
            oov_token="<OOV>",
            filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
        )

        self.tokenizer.fit_on_texts(sentences)
        self.vocab_size = min(self.vocab_size, len(self.tokenizer.word_index) + 1)

        sequences = self.create_sequences(sentences)

        if not sequences:
            raise ValueError("No valid sequences created. Please provide more text.")

        sequences = pad_sequences(
            sequences, maxlen=self.max_seq_length, padding="pre", truncating="pre"
        )

        X = sequences[:, :-1]
        y = sequences[:, -1]
        y = tf.keras.utils.to_categorical(y, num_classes=self.vocab_size)

        return X, y

    def build_model(self):
        """Build optimized model for Streamlit"""
        self.model = Sequential(
            [
                Embedding(
                    self.vocab_size,
                    self.embedding_dim,
                    input_length=self.max_seq_length - 1,
                    mask_zero=True,
                ),
                Bidirectional(
                    LSTM(
                        self.lstm_units,
                        return_sequences=True,
                        dropout=self.dropout_rate,
                    )
                ),
                LayerNormalization(),
                Bidirectional(LSTM(self.lstm_units // 2, dropout=self.dropout_rate)),
                Dense(256, activation="relu"),
                Dropout(self.dropout_rate),
                Dense(self.vocab_size, activation="softmax"),
            ]
        )

        optimizer = AdamW(learning_rate=0.001, weight_decay=0.01)
        self.model.compile(
            loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"]
        )

        return self.model

    def train_with_progress(self, X, y, epochs=15, batch_size=32, validation_split=0.2):
        """Train model with Streamlit progress tracking"""

        # Create progress bars
        progress_bar = st.progress(0)
        status_text = st.empty()
        metrics_placeholder = st.empty()

        # Custom callback for Streamlit
        class StreamlitCallback(tf.keras.callbacks.Callback):
            def __init__(
                self, total_epochs, progress_bar, status_text, metrics_placeholder
            ):
                self.total_epochs = total_epochs
                self.progress_bar = progress_bar
                self.status_text = status_text
                self.metrics_placeholder = metrics_placeholder
                self.epoch_losses = []
                self.epoch_accuracies = []

            def on_epoch_end(self, epoch, logs=None):
                # Update progress
                progress = (epoch + 1) / self.total_epochs
                self.progress_bar.progress(progress)

                # Update status
                self.status_text.text(
                    f"Epoch {epoch + 1}/{self.total_epochs} - "
                    f'Loss: {logs["loss"]:.4f} - '
                    f'Accuracy: {logs["accuracy"]:.4f}'
                )

                # Store metrics
                self.epoch_losses.append(logs["loss"])
                self.epoch_accuracies.append(logs["accuracy"])

                # Update metrics display
                col1, col2 = self.metrics_placeholder.columns(2)
                with col1:
                    st.metric("Current Loss", f"{logs['loss']:.4f}")
                with col2:
                    st.metric("Current Accuracy", f"{logs['accuracy']:.4f}")

        # Train model
        callback = StreamlitCallback(
            epochs, progress_bar, status_text, metrics_placeholder
        )

        history = self.model.fit(
            X,
            y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=[
                callback,
                EarlyStopping(
                    monitor="val_loss", patience=5, restore_best_weights=True
                ),
            ],
            verbose=0,
        )

        # Clear progress indicators
        time.sleep(1)
        progress_bar.empty()
        status_text.empty()
        metrics_placeholder.empty()

        return history, callback.epoch_losses, callback.epoch_accuracies

    def predict_next_words(self, seed_text, n_words=5, temperature=0.8, top_k=10):
        """Predict next words with temperature sampling"""
        if not self.model or not self.tokenizer:
            return "Model not trained yet!"

        result = seed_text.lower()

        for _ in range(n_words):
            token_list = self.tokenizer.texts_to_sequences([result])[0]
            token_list = pad_sequences(
                [token_list], maxlen=self.max_seq_length - 1, padding="pre"
            )

            predictions = self.model.predict(token_list, verbose=0)[0]

            # Apply temperature
            predictions = np.log(predictions + 1e-10) / temperature
            predictions = np.exp(predictions)
            predictions = predictions / np.sum(predictions)

            # Top-k sampling
            top_k_indices = np.argsort(predictions)[-top_k:]
            top_k_probs = predictions[top_k_indices]
            top_k_probs = top_k_probs / np.sum(top_k_probs)

            predicted_id = np.random.choice(top_k_indices, p=top_k_probs)

            # Find word
            output_word = ""
            for word, index in self.tokenizer.word_index.items():
                if index == predicted_id:
                    output_word = word
                    break

            if output_word and output_word != "<OOV>":
                result += " " + output_word
            else:
                break

        return result

    def get_word_probabilities(self, seed_text, top_n=10):
        """Get top word probabilities"""
        if not self.model or not self.tokenizer:
            return []

        token_list = self.tokenizer.texts_to_sequences([seed_text.lower()])[0]
        token_list = pad_sequences(
            [token_list], maxlen=self.max_seq_length - 1, padding="pre"
        )

        predictions = self.model.predict(token_list, verbose=0)[0]
        top_indices = np.argsort(predictions)[-top_n:][::-1]

        results = []
        for idx in top_indices:
            for word, word_idx in self.tokenizer.word_index.items():
                if word_idx == idx:
                    results.append((word, predictions[idx]))
                    break

        return results


# Initialize session state
if "predictor" not in st.session_state:
    st.session_state.predictor = StreamlitNLPPredictor()
if "model_trained" not in st.session_state:
    st.session_state.model_trained = False
if "training_history" not in st.session_state:
    st.session_state.training_history = None

# Main App UI
st.markdown('<h1 class="main-header">🤖 AI Text Predictor</h1>', unsafe_allow_html=True)

# Sidebar for model configuration
st.sidebar.header("⚙️ Model Configuration")

vocab_size = st.sidebar.slider("Vocabulary Size", 1000, 10000, 5000, 500)
embedding_dim = st.sidebar.slider("Embedding Dimension", 64, 256, 128, 32)
lstm_units = st.sidebar.slider("LSTM Units", 64, 256, 128, 32)
max_seq_length = st.sidebar.slider("Max Sequence Length", 10, 50, 30, 5)
dropout_rate = st.sidebar.slider("Dropout Rate", 0.1, 0.5, 0.3, 0.1)

# Update predictor if parameters changed
current_config = (vocab_size, embedding_dim, lstm_units, max_seq_length, dropout_rate)
if (
    "last_config" not in st.session_state
    or st.session_state.last_config != current_config
):
    st.session_state.predictor = StreamlitNLPPredictor(
        vocab_size, embedding_dim, lstm_units, max_seq_length, dropout_rate
    )
    st.session_state.model_trained = False
    st.session_state.last_config = current_config

# Main content
tab1, tab2, tab3, tab4 = st.tabs(
    ["📝 Train Model", "🎯 Generate Text", "📊 Analytics", "💾 Model Info"]
)

with tab1:
    st.header("Train Your Text Prediction Model")

    # Text input options
    input_method = st.radio(
        "Choose input method:", ["Upload Text File", "Paste Text", "Use Sample Text"]
    )

    text_data = ""

    if input_method == "Upload Text File":
        uploaded_file = st.file_uploader("Choose a text file", type=["txt"])
        if uploaded_file:
            text_data = str(uploaded_file.read(), "utf-8")
            st.success(f"File uploaded! Text length: {len(text_data)} characters")

    elif input_method == "Paste Text":
        text_data = st.text_area(
            "Paste your text here:",
            height=200,
            placeholder="Enter your training text here...",
        )

    else:  # Sample text
        text_data = (
            """
        Sherlock Holmes was a brilliant consulting detective who lived at 221B Baker Street in London. 
        He was known for his extraordinary deductive abilities and scientific approach to solving crimes.
        Dr. John Watson was his loyal friend and companion who documented their adventures.
        Together they solved many mysterious cases that baffled Scotland Yard.
        Holmes had a sharp mind and could observe details that others missed completely.
        His methods were revolutionary for the Victorian era and involved careful observation.
        The detective could deduce a person's entire life story from small clues.
        Watson admired Holmes' brilliant mind and logical reasoning abilities.
        They lived together and solved crimes throughout London and beyond.
        Holmes played the violin and had extensive knowledge of chemistry and forensics.
        """
            * 10
        )  # Repeat for more training data

        st.info("Using sample Sherlock Holmes text for demonstration")

    # Training parameters
    col1, col2 = st.columns(2)
    with col1:
        epochs = st.slider("Training Epochs", 5, 50, 15, 5)
        batch_size = st.selectbox("Batch Size", [16, 32, 64, 128], index=1)
    with col2:
        validation_split = st.slider("Validation Split", 0.1, 0.3, 0.2, 0.05)

    # Train button
    if st.button("🚀 Train Model", type="primary"):
        if text_data.strip():
            try:
                with st.spinner("Preparing data..."):
                    X, y = st.session_state.predictor.prepare_data(text_data)
                    st.session_state.predictor.build_model()

                st.info(
                    f"Training on {X.shape[0]} sequences with vocabulary of {st.session_state.predictor.vocab_size} words"
                )

                # Train with progress tracking
                history, losses, accuracies = (
                    st.session_state.predictor.train_with_progress(
                        X, y, epochs, batch_size, validation_split
                    )
                )

                st.session_state.model_trained = True
                st.session_state.training_history = {
                    "loss": losses,
                    "accuracy": accuracies,
                }

                st.success("🎉 Model trained successfully!")
                st.balloons()

            except Exception as e:
                st.error(f"Training failed: {str(e)}")
        else:
            st.warning("Please provide training text!")

with tab2:
    st.header("Generate Text Predictions")

    if not st.session_state.model_trained:
        st.warning("⚠️ Please train the model first in the 'Train Model' tab!")
    else:
        # Text generation interface
        col1, col2 = st.columns([2, 1])

        with col1:
            seed_text = st.text_input(
                "Enter seed text:",
                value="sherlock holmes",
                placeholder="Enter starting text...",
            )

        with col2:
            n_words = st.slider("Words to generate", 1, 20, 5)

        # Generation parameters
        col1, col2, col3 = st.columns(3)
        with col1:
            temperature = st.slider("Creativity (Temperature)", 0.1, 2.0, 0.8, 0.1)
        with col2:
            top_k = st.slider("Top-K Sampling", 5, 50, 10, 5)
        with col3:
            num_generations = st.slider("Number of Variations", 1, 5, 3)

        if st.button("✨ Generate Text", type="primary"):
            if seed_text.strip():
                st.subheader("Generated Text:")

                for i in range(num_generations):
                    result = st.session_state.predictor.predict_next_words(
                        seed_text, n_words, temperature, top_k
                    )

                    st.markdown(
                        f"""
                    <div class="prediction-box">
                        <strong>Variation {i+1}:</strong><br>
                        {result}
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )
            else:
                st.warning("Please enter seed text!")

        # Word probabilities
        if st.checkbox("Show Next Word Probabilities"):
            if seed_text.strip():
                probs = st.session_state.predictor.get_word_probabilities(seed_text, 15)
                if probs:
                    df = pd.DataFrame(probs, columns=["Word", "Probability"])

                    # Create bar chart
                    fig = px.bar(
                        df,
                        x="Word",
                        y="Probability",
                        title=f'Most Likely Next Words for: "{seed_text}"',
                    )
                    fig.update_layout(xaxis_tickangle=45)
                    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.header("Training Analytics")

    if st.session_state.training_history:
        # Training metrics
        col1, col2 = st.columns(2)

        with col1:
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            final_loss = st.session_state.training_history["loss"][-1]
            st.metric("Final Training Loss", f"{final_loss:.4f}")
            st.markdown("</div>", unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            final_acc = st.session_state.training_history["accuracy"][-1]
            st.metric("Final Training Accuracy", f"{final_acc:.4f}")
            st.markdown("</div>", unsafe_allow_html=True)

        # Training curves
        fig = go.Figure()

        epochs_range = list(
            range(1, len(st.session_state.training_history["loss"]) + 1)
        )

        fig.add_trace(
            go.Scatter(
                x=epochs_range,
                y=st.session_state.training_history["loss"],
                mode="lines+markers",
                name="Training Loss",
                line=dict(color="red"),
            )
        )

        fig2 = go.Figure()
        fig2.add_trace(
            go.Scatter(
                x=epochs_range,
                y=st.session_state.training_history["accuracy"],
                mode="lines+markers",
                name="Training Accuracy",
                line=dict(color="blue"),
            )
        )

        col1, col2 = st.columns(2)
        with col1:
            fig.update_layout(title="Training Loss Over Time")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig2.update_layout(title="Training Accuracy Over Time")
            st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("No training data available. Train a model to see analytics!")

with tab4:
    st.header("Model Information")

    if st.session_state.model_trained:
        # Model architecture
        st.subheader("Model Architecture")

        col1, col2 = st.columns(2)
        with col1:
            st.info(f"**Vocabulary Size:** {st.session_state.predictor.vocab_size:,}")
            st.info(f"**Embedding Dimension:** {embedding_dim}")
            st.info(f"**LSTM Units:** {lstm_units}")

        with col2:
            st.info(f"**Max Sequence Length:** {max_seq_length}")
            st.info(f"**Dropout Rate:** {dropout_rate}")

            # Calculate approximate model size
            total_params = st.session_state.predictor.model.count_params()
            st.info(f"**Total Parameters:** {total_params:,}")

        # Model summary
        if st.checkbox("Show Detailed Model Summary"):
            # Capture model summary
            stream = io.StringIO()
            st.session_state.predictor.model.summary(
                print_fn=lambda x: stream.write(x + "\n")
            )
            summary_string = stream.getvalue()
            st.text(summary_string)

        # Download model option
        st.subheader("Model Export")
        if st.button("📥 Prepare Model for Download"):
            # Note: Actual model saving would require more complex implementation
            st.info(
                "Model export functionality would be implemented here for production use"
            )

    else:
        st.info("Train a model to see detailed information!")

# Footer
st.markdown("---")
st.markdown(
    """
<div style='text-align: center; color: #888;'>
    |  NLP Text Prediction
</div>
""",
    unsafe_allow_html=True,
)
