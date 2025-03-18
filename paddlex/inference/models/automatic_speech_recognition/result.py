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

from typing import List, Dict, Any
import json
import numpy as np


class Result:
    """ASR Result Container for ChunkConformer Model"""

    def __init__(self, transcript: str, chunk_results: List[Dict[str, Any]] = None):
        self.transcript = transcript
        self.chunk_results = chunk_results or []
        self.metadata = {
            "model_type": "chunk_conformer",
            "decoding_method": "ctc_greedy",
        }

    def add_chunk_result(
        self, chunk_idx: int, partial_text: str, confidence: float, features: np.ndarray
    ):
        """Store intermediate chunk processing results"""
        self.chunk_results.append(
            {
                "chunk_index": chunk_idx,
                "partial_text": partial_text,
                "confidence": round(float(confidence), 4),
                "feature_shape": features.shape,
            }
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary format"""
        return {
            "transcript": self.transcript,
            "chunk_count": len(self.chunk_results),
            "chunk_details": self.chunk_results,
            "metadata": self.metadata,
        }

    def to_json(self) -> str:
        """Serialize result to JSON string"""
        return json.dumps(self.to_dict(), indent=2, default=self._json_serializer)

    def _json_serializer(self, obj):
        """Custom JSON serializer for numpy types"""
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    def __str__(self) -> str:
        return self.transcript

    def __repr__(self) -> str:
        return f"<ASRResult transcript='{self.transcript[:50]}...' chunks={len(self.chunk_results)}>"


class ResultBuilder:
    """Factory for constructing ASR results"""

    @staticmethod
    def from_continuous_output(transcript: str, chunk_stride: int):
        """Create result from continuous decoding output"""
        # Implement logic to split transcript into chunks based on stride
        return Result(transcript)

    @classmethod
    def from_chunked_output(cls, chunks: List[Dict[str, Any]]):
        """Create result from pre-chunked outputs"""
        full_transcript = " ".join([chunk["text"] for chunk in chunks])
        result = cls(full_transcript)
        result.chunk_results = chunks
        return result
