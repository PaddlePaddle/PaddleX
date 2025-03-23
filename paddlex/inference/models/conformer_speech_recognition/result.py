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
import os
from ...common.result import BaseResult, StrMixin, JsonMixin


class ConformerSpeechResult(BaseResult):
    """ASR Result Container for Conformer Speech Recognition Model"""

    def __init__(self, data: dict) -> None:
        """
        Initialize the ConformerSpeechResult.

        Args:
            data (dict): The initial data containing transcript and optional chunk results.
                Must contain 'transcript' key.

        Raises:
            AssertionError: If the required key ('transcript') is not found in the data.
        """
        super().__init__(data)
        assert "transcript" in self.keys(), "transcript is not found in the data"

        # Initialize chunk_results if not present
        if "chunk_results" not in self.keys():
            self["chunk_results"] = []

        # Set default metadata
        if "metadata" not in self.keys():
            self["metadata"] = {
                "model_type": "conformer_speech",
                "decoding_method": "ctc_greedy",
            }

    def add_chunk_result(
        self, chunk_idx: int, partial_text: str, confidence: float, features: np.ndarray
    ):
        """
        Store intermediate chunk processing results.

        Args:
            chunk_idx: Index of the current chunk.
            partial_text: Recognized text for this chunk.
            confidence: Confidence score for the recognition.
            features: Feature matrix used for this chunk.
        """
        if "chunk_results" not in self.keys():
            self["chunk_results"] = []

        self["chunk_results"].append(
            {
                "chunk_index": chunk_idx,
                "partial_text": partial_text,
                "confidence": round(float(confidence), 4),
                "feature_shape": features.shape if hasattr(features, "shape") else None,
            }
        )

    def visualize(self, save_path: str = None, show: bool = False) -> None:
        """
        Visualize or save the speech recognition results.

        Args:
            save_path: Path to save the results. If None, results are not saved.
            show: Whether to display the results.
        """
        # Create a formatted output of the transcript
        result_text = f"Transcript: {self['transcript']}\n"
        result_text += f"Confidence: {self.get_average_confidence():.4f}\n"
        result_text += f"Chunks: {len(self['chunk_results'])}\n"

        if "audio_path" in self.keys():
            result_text += f"Audio source: {self['audio_path']}\n"

        # Save results to file if path is provided
        if save_path is not None:
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            # Save text results
            with open(os.path.join(save_path, "transcript.txt"), "w") as f:
                f.write(result_text)

            # Save full result as JSON
            with open(os.path.join(save_path, "result.json"), "w") as f:
                json.dump(self.to_dict(), f, indent=2, default=self._json_serializer)

        # Display results if requested
        if show:
            print("\n" + "=" * 50)
            print("SPEECH RECOGNITION RESULTS")
            print("=" * 50)
            print(result_text)
            print("=" * 50 + "\n")

        return

    def get_average_confidence(self) -> float:
        """
        Calculate the average confidence across all chunks.

        Returns:
            Average confidence score or 0.0 if no chunks available.
        """
        if not self["chunk_results"]:
            return 0.0

        confidences = [chunk.get("confidence", 0.0) for chunk in self["chunk_results"]]
        return sum(confidences) / len(confidences) if confidences else 0.0

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert result to dictionary format.

        Returns:
            Dictionary representation of the result.
        """
        return {
            "transcript": self["transcript"],
            "chunk_count": len(self["chunk_results"]),
            "chunk_details": self["chunk_results"],
            "metadata": self["metadata"],
        }

    def _json_serializer(self, obj):
        """
        Custom JSON serializer for numpy types.

        Args:
            obj: Object to serialize.

        Returns:
            JSON serializable version of the object.
        """
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    def __str__(self) -> str:
        """String representation showing the transcript."""
        return self["transcript"]

    def __repr__(self) -> str:
        """Detailed string representation for debugging."""
        transcript_preview = (
            self["transcript"][:50] + "..."
            if len(self["transcript"]) > 50
            else self["transcript"]
        )
        return f"<ConformerSpeechResult transcript='{transcript_preview}' chunks={len(self['chunk_results'])}>"


class ResultBuilder:
    """Factory for constructing ASR results"""

    @staticmethod
    def from_continuous_output(transcript: str, chunk_stride: int = None):
        """
        Create result from continuous decoding output.

        Args:
            transcript: The complete transcript text.
            chunk_stride: Optional stride used for chunking.

        Returns:
            ConformerSpeechResult object.
        """
        data = {
            "transcript": transcript,
            "metadata": {
                "model_type": "conformer_speech",
                "decoding_method": "ctc_greedy",
                "chunk_stride": chunk_stride,
            },
        }
        return ConformerSpeechResult(data)

    @staticmethod
    def from_chunked_output(chunks: List[Dict[str, Any]]):
        """
        Create result from pre-chunked outputs.

        Args:
            chunks: List of chunk dictionaries with recognition results.

        Returns:
            ConformerSpeechResult object.
        """
        full_transcript = " ".join(
            [chunk.get("text", "") for chunk in chunks if "text" in chunk]
        )
        data = {
            "transcript": full_transcript,
            "chunk_results": chunks,
            "metadata": {
                "model_type": "conformer_speech",
                "decoding_method": "ctc_greedy",
                "chunked_processing": True,
            },
        }
        return ConformerSpeechResult(data)
