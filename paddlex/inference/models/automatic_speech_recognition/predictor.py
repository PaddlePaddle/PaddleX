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

import numpy as np
from pathlib import Path
from typing import Optional, Union
from .processors import Preprocess, Postprocess
from ..base import BasicPredictor

class ASRPredictor(BasicPredictor):
    """ChunkConformer Automatic Speech Recognition Predictor"""
    
    def __init__(self, model_dir: Union[str, Path], config: dict, device: Optional[str] = None, **kwargs):
        super().__init__(model_dir, config, device, **kwargs)
        self.sample_rate = config.get('sample_rate', 16000)
        
        # Initialize model first
        self.model = self.create_model()
        
        # Chunk processing config
        self.chunk_size = config.get('chunk_size', 16)  # in seconds
        self.stride = config.get('stride', 4)         # in seconds
        
        # Audio feature extractor
        self.feature_extractor = Preprocess(
            sample_rate=self.sample_rate,
            n_fft=400,
            hop_length=160
        )
        
        # Text decoder with state tracking
        self.decoder = Postprocess(
            vocab_path=Path(model_dir)/'vocab.txt',
            decoding_method='ctc_greedy',
            chunk_stride=self.stride
        )

    def preprocess(self, audio_path: Union[str, Path]):
        """Process audio input into features"""
        return self.feature_extractor(audio_path)

    def postprocess(self, model_outputs: np.ndarray, decoder_state: Optional[dict] = None):
        """Decode model outputs to text with state management"""
        return self.decoder(model_outputs, decoder_state)

    def predict(self, audio_path: Union[str, Path]):
        """Streaming prediction with chunk processing"""
        full_transcript = []
        decoder_state = None
        
        # Process audio in chunks with overlap
        for chunk_idx, audio_chunk in enumerate(self.load_audio_chunks(audio_path)):
            # Extract features for current chunk
            features = self.feature_extractor(audio_chunk)
            
            # Run model inference
            chunk_outputs = self.model_infer(features)
            
            # Decode with state passing between chunks
            text, decoder_state = self.postprocess(chunk_outputs, decoder_state)
            
            # Store intermediate results
            if chunk_idx > 0 and self.stride > 0:
                # Remove overlapping part from previous chunk
                full_transcript = full_transcript[:-self.stride]
                
            full_transcript.extend(text)
            
        return ''.join(full_transcript)

    def load_audio_chunks(self, audio_path: Union[str, Path]):
        """Yield audio chunks with configurable size and stride"""
        import soundfile as sf
        
        # Load full audio
        audio, sr = sf.read(audio_path)
        if sr != self.sample_rate:
            raise ValueError(f"Audio sample rate {sr}Hz doesn't match model rate {self.sample_rate}Hz")
            
        # Convert to mono if needed
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
            
        # Calculate chunk parameters in samples
        chunk_samples = int(self.chunk_size * self.sample_rate)
        stride_samples = int(self.stride * self.sample_rate)
        
        # Split audio into overlapping chunks
        for start in range(0, len(audio), chunk_samples - stride_samples):
            end = start + chunk_samples
            chunk = audio[start:end]
            
            # Pad last chunk if needed
            if len(chunk) < chunk_samples:
                chunk = np.pad(chunk, (0, chunk_samples - len(chunk)))
                
            yield chunk
