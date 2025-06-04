"""
PaddleX Conformer model implementation aligned with PaddleSpeech.
This module provides the Conformer architecture for speech recognition tasks.
"""

from .model import ConformerModel
from .trainer import ConformerTrainer
from .predictor import ConformerPredictor

__all__ = ['ConformerModel', 'ConformerTrainer', 'ConformerPredictor']
