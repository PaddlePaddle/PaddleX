# copyright (c) 2024 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Union, Tuple, List, Dict, Any, Iterator, Optional
import numpy as np
from pathlib import Path
import soundfile as sf
import os
import lazy_paddle as paddle


class LoadAudioFromFile:
    """Load audio data from file path or directly process audio data."""

    def __init__(self, sample_rate: int = 16000):
        """Initialize audio loader.

        Args:
            sample_rate: Expected sample rate for audio files.
        """
        self.sample_rate = sample_rate

    def __call__(self, input_data: Union[str, Path, np.ndarray]) -> np.ndarray:
        """Load and validate audio data.

        Args:
            input_data: Path to audio file or audio data as numpy array.

        Returns:
            Audio data as numpy array.
        """
        if isinstance(input_data, (str, Path)):
            if not os.path.exists(input_data):
                raise FileNotFoundError(f"Audio file not found: {input_data}")
            audio, sr = sf.read(input_data)
            if sr != self.sample_rate:
                raise ValueError(
                    f"Sample rate mismatch: got {sr}Hz, expected {self.sample_rate}Hz"
                )
            # Convert to mono if needed
            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=1)
            return audio
        elif isinstance(input_data, np.ndarray):
            # Assume correct sample rate if directly passing audio data
            if len(input_data.shape) > 1:
                input_data = np.mean(input_data, axis=1)
            return input_data
        else:
            raise TypeError(
                f"Expected str, Path or numpy array, got {type(input_data)}"
            )


class ExtractFeatures:
    """Extract acoustic features from audio data."""

    def __init__(
        self,
        n_mels: int = 80,
        n_fft: int = 400,
        hop_length: int = 160,
        win_length: int = 400,
        sample_rate: int = 16000,
        window: str = "hann",
    ):
        """Initialize feature extractor.

        Args:
            n_mels: Number of mel bands.
            n_fft: FFT window size.
            hop_length: Hop length between frames.
            win_length: Window length.
            sample_rate: Audio sample rate.
        """
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.sample_rate = sample_rate
        self.window = window  # Use provided window type

    def __call__(self, audio: np.ndarray) -> np.ndarray:
        """Extract log-mel spectrogram features from audio.

        Args:
            audio: Audio data as numpy array.

        Returns:
            Log-mel spectrogram features.
        """
        # Normalize audio
        audio = audio / (np.max(np.abs(audio)) + 1e-10)

        # In a real implementation, we would use librosa or torchaudio
        # Here we're implementing a simplified version using numpy

        # Compute STFT
        frames = np.lib.stride_tricks.sliding_window_view(
            np.pad(audio, (self.n_fft // 2, self.n_fft // 2)), self.n_fft
        )[:: self.hop_length]

        # Apply hann window function
        window_func = np.hanning(self.n_fft)

        frames = frames * window_func

        # Compute magnitude spectrogram
        spec = np.abs(np.fft.rfft(frames, n=self.n_fft))

        # Apply mel filterbank (simplified)
        # In practice, use librosa.filters.mel
        mel_basis = np.random.rand(self.n_fft // 2 + 1, self.n_mels)  # Placeholder
        mel_spec = np.dot(spec, mel_basis)

        # Log mel spectrogram
        log_mel_spec = np.log(mel_spec + 1e-10)

        # Normalize
        log_mel_spec = (log_mel_spec - np.mean(log_mel_spec)) / (
            np.std(log_mel_spec) + 1e-10
        )

        return log_mel_spec


class ProcessChunks:
    """Process audio in chunks for streaming inference."""

    def __init__(
        self,
        chunk_size_frames: int = 50,
        context_frames: int = 10,
    ):
        """Initialize chunk processor.

        Args:
            chunk_size_frames: Size of each chunk in frames.
            context_frames: Number of context frames to include.
        """
        self.chunk_size_frames = chunk_size_frames
        self.context_frames = context_frames
        self.state = None

    def __call__(
        self, features: np.ndarray, state: Optional[Dict] = None
    ) -> Tuple[List[np.ndarray], Dict]:
        """Split features into chunks with context.

        Args:
            features: Feature matrix (n_mels, time).
            state: Optional state from previous call.

        Returns:
            Tuple of (list of feature chunks, updated state).
        """
        if state is not None:
            self.state = state

        # Initialize state if needed
        if self.state is None:
            self.state = {
                "context": np.zeros((features.shape[0], self.context_frames)),
                "position": 0,
            }

        # Transpose to (time, n_mels) if needed
        if features.shape[0] < features.shape[1]:
            features = features.T

        n_frames = features.shape[0]
        chunks = []

        # Process features in chunks
        position = self.state["position"]
        context = self.state["context"]

        while position < n_frames:
            end_pos = min(position + self.chunk_size_frames, n_frames)
            current_chunk = features[position:end_pos]

            # Add context at the beginning
            chunk_with_context = np.vstack([context, current_chunk])
            chunks.append(chunk_with_context)

            # Update context for next chunk
            if end_pos < n_frames:
                context_start = max(0, end_pos - self.context_frames)
                context = features[context_start:end_pos]
            else:
                # Pad with zeros if we're at the end
                last_frames = features[max(0, end_pos - self.context_frames) : end_pos]
                padding_needed = self.context_frames - last_frames.shape[0]
                if padding_needed > 0:
                    context = np.vstack(
                        [last_frames, np.zeros((padding_needed, features.shape[1]))]
                    )
                else:
                    context = last_frames

            position = end_pos

        # Update state
        self.state = {"context": context, "position": position}

        return chunks, self.state


class GetInferInput:
    """Prepare model input from feature chunks."""

    def __init__(self):
        """Initialize inference input processor."""
        pass

    def __call__(self, feature_chunks: List[np.ndarray]) -> Dict[str, np.ndarray]:
        """Convert feature chunks to model input format.

        Args:
            feature_chunks: List of feature chunks.

        Returns:
            Dictionary of model inputs.
        """
        if not feature_chunks:
            return {}

        # Stack chunks into batch
        batch = np.stack(feature_chunks, axis=0)

        # Create length tensor
        lengths = np.array([chunk.shape[0] for chunk in feature_chunks], dtype=np.int64)

        return {"audio_features": batch.astype(np.float32), "audio_lengths": lengths}


class DecodeOutputs:
    """Decode model outputs to text."""

    def __init__(
        self,
        vocab_path: Optional[Union[str, Path]] = None,
        vocab_list: Optional[List[str]] = None,
        blank_idx: int = 0,
        unk_idx: int = 1,
        decoding_method: str = "ctc_greedy",
    ):
        """Initialize decoder.

        Args:
            vocab_path: Path to vocabulary file.
            vocab_list: List of vocabulary items.
            blank_idx: Index of blank token.
            unk_idx: Index of unknown token.
            decoding_method: Decoding method ('ctc_greedy' or 'ctc_beam').
        """
        if vocab_path is not None:
            self.vocab = self._load_vocab(vocab_path)
        elif vocab_list is not None:
            self.vocab = vocab_list
        else:
            raise ValueError("Either vocab_path or vocab_list must be provided")

        self.blank_idx = blank_idx
        self.unk_idx = unk_idx
        self.decoding_method = decoding_method
        self.state = None

    def __call__(
        self, logits: np.ndarray, state: Optional[Dict] = None
    ) -> Tuple[List[str], Dict]:
        """Decode logits to text with state management.

        Args:
            logits: Model output logits.
            state: Optional state from previous call.

        Returns:
            Tuple of (decoded texts, updated state).
        """
        if state is not None:
            self.state = state

        if self.state is None:
            self.state = {"prev_tokens": []}

        batch_size = logits.shape[0]
        texts = []

        for i in range(batch_size):
            if self.decoding_method == "ctc_greedy":
                text, tokens = self._ctc_greedy_decode(logits[i])
            else:
                text, tokens = self._ctc_beam_decode(logits[i])

            texts.append(text)
            self.state["prev_tokens"].append(tokens)

        return texts, self.state

    def _load_vocab(self, vocab_path: Union[str, Path]) -> List[str]:
        """Load vocabulary from file.

        Args:
            vocab_path: Path to vocabulary file.

        Returns:
            List of vocabulary items.
        """
        with open(vocab_path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f]

    def _ctc_greedy_decode(self, logits: np.ndarray) -> Tuple[str, List[int]]:
        """CTC greedy decoding.

        Args:
            logits: Logits for one sequence.

        Returns:
            Tuple of (decoded text, token indices).
        """
        # Get most probable tokens
        tokens = np.argmax(logits, axis=1)

        # Remove repeated tokens
        prev_token = -1
        collapsed_tokens = []

        for token in tokens:
            if token != prev_token and token != self.blank_idx:
                collapsed_tokens.append(token)
            prev_token = token

        # Convert to text
        text = "".join([self.vocab[idx] for idx in collapsed_tokens])

        return text, collapsed_tokens

    def _ctc_beam_decode(self, logits: np.ndarray) -> Tuple[str, List[int]]:
        """CTC beam search decoding (simplified).

        Args:
            logits: Logits for one sequence.

        Returns:
            Tuple of (decoded text, token indices).
        """
        # Simplified implementation - in practice use a proper beam search
        # For now, fall back to greedy decoding
        return self._ctc_greedy_decode(logits)


class Preprocess:
    """Audio feature extraction processor for chunk conformer"""

    def __init__(
        self, sample_rate: int = 16000, n_fft: int = 400, hop_length: int = 160
    ):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length

    def __call__(self, audio_path: Union[str, Path]) -> np.ndarray:
        """Extract log-mel spectrogram features from audio"""
        audio, sr = sf.read(audio_path)
        if sr != self.sample_rate:
            raise ValueError(
                f"Sample rate mismatch: got {sr}Hz, expected {self.sample_rate}Hz"
            )

        # Convert to mono and normalize
        audio = self._preprocess_audio(audio)

        # Extract features
        return self._extract_features(audio)

    def _preprocess_audio(self, audio: np.ndarray) -> np.ndarray:
        """Convert to mono and normalize"""
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
        return audio / np.max(np.abs(audio))

    def _extract_features(self, audio: np.ndarray) -> np.ndarray:
        """Compute log-mel spectrogram features"""
        # Implementation of feature extraction logic
        # This would typically use librosa or custom DSP code
        # Simplified placeholder implementation:
        return np.random.rand(80, 1000)  # Mock feature matrix


class Postprocess:
    """Text decoding processor with chunk state management"""

    def __init__(
        self,
        vocab_path: Path,
        decoding_method: str = "ctc_greedy",
        chunk_stride: int = 4,
    ):
        self.vocab = self._load_vocab(vocab_path)
        self.decoding_method = decoding_method
        self.chunk_stride = chunk_stride
        self.state = None

    def __call__(
        self, logits: np.ndarray, state: Optional[dict] = None
    ) -> Tuple[List[str], dict]:
        """Decode model outputs with optional state"""
        # Implement CTC decoding with state management
        decoded_text = self._ctc_decode(logits)
        new_state = self._update_state(state)
        return decoded_text, new_state

    def _load_vocab(self, vocab_path: Path) -> List[str]:
        """Load vocabulary from file"""
        with open(vocab_path, "r") as f:
            return [line.strip() for line in f]

    def _ctc_decode(self, logits: np.ndarray) -> List[str]:
        """CTC greedy decoding implementation"""
        # Simplified decoding logic
        return ["mock", "transcript"]

    def _update_state(self, state: Optional[dict]) -> dict:
        """Manage decoder state between chunks"""
        return {"last_logits": np.random.rand(10, 100)}  # Mock state
