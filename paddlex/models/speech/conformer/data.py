"""
Data processing module for Conformer model in PaddleX.

This module provides utilities for loading, preprocessing, and batching speech data for
training and inference with Conformer models.
"""

import os
import numpy as np
import paddle
import librosa


class AudioDataset(paddle.io.Dataset):
    """
    Dataset class for audio data processing.
    
    Args:
        data_list (list or str): List of audio file paths and corresponding transcriptions,
                                 or path to a manifest file.
        config (dict): Data processing configuration.
        mode (str): 'train', 'dev', or 'test'.
    """
    
    def __init__(self, data_list, config, mode='train'):
        super(AudioDataset, self).__init__()
        
        self.mode = mode
        self.config = config
        self.sample_rate = config.get('sample_rate', 16000)
        self.feature_type = config.get('feature_type', 'fbank')
        self.n_mels = config.get('n_mels', 80)
        self.n_fft = config.get('n_fft', 512)
        self.win_length = config.get('win_length', 400)
        self.hop_length = config.get('hop_length', 160)
        
        # Load data list
        self.data_list = self._load_data_list(data_list)
        
        # Set up augmentation if in training mode
        self.spec_augment = config.get('spec_augment', False) and mode == 'train'
        self.spec_aug_config = config.get('spec_aug_config', {})
    
    def _load_data_list(self, data_list):
        """
        Load data list from manifest file or directly use provided list.
        
        Args:
            data_list (list or str): List of samples or path to manifest file.
            
        Returns:
            list: List of samples with audio path and transcription.
        """
        if isinstance(data_list, str) and os.path.isfile(data_list):
            samples = []
            with open(data_list, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        # Expected format: {"audio_filepath": "...", "text": "..."}
                        try:
                            import json
                            data = json.loads(line)
                            samples.append({
                                'audio_path': data['audio_filepath'],
                                'text': data['text']
                            })
                        except:
                            continue
            return samples
        else:
            return data_list
    
    def _extract_features(self, audio_path):
        """
        Extract features from audio file.
        
        Args:
            audio_path (str): Path to the audio file.
            
        Returns:
            np.ndarray: Extracted features.
        """
        try:
            # Load audio
            waveform, _ = librosa.load(audio_path, sr=self.sample_rate)
            
            # Extract features based on feature type
            if self.feature_type == 'fbank':
                features = librosa.feature.melspectrogram(
                    y=waveform,
                    sr=self.sample_rate,
                    n_fft=self.n_fft,
                    win_length=self.win_length,
                    hop_length=self.hop_length,
                    n_mels=self.n_mels
                )
                features = np.log(features + 1e-6)
                features = features.T  # (Time, Freq)
            else:
                # Default to MFCC if feature type not recognized
                features = librosa.feature.mfcc(
                    y=waveform,
                    sr=self.sample_rate,
                    n_mfcc=self.n_mels
                )
                features = features.T  # (Time, Freq)
            
            # Apply normalization
            features = (features - np.mean(features)) / np.std(features)
            
            return features
        except Exception as e:
            print(f"Error processing {audio_path}: {str(e)}")
            return np.zeros((100, self.n_mels))  # Return dummy features
    
    def _apply_spec_augment(self, features):
        """
        Apply SpecAugment for data augmentation.
        
        Args:
            features (np.ndarray): Input features.
            
        Returns:
            np.ndarray: Augmented features.
        """
        # TODO: Implement SpecAugment augmentation
        return features
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        """Get a sample from the dataset."""
        sample = self.data_list[idx]
        
        # Extract features
        features = self._extract_features(sample['audio_path'])
        
        # Apply augmentation if required
        if self.spec_augment:
            features = self._apply_spec_augment(features)
        
        # Process text
        text = sample.get('text', '')
        # TODO: Implement text processing (tokenization)
        
        return features, text


class AudioCollator:
    """
    Collator class for batching audio samples.
    
    Handles padding and stacking of variable-length sequences.
    """
    
    def __init__(self, padding_idx=0):
        self.padding_idx = padding_idx
    
    def __call__(self, batch):
        """
        Process a batch of samples.
        
        Args:
            batch (list): List of (features, text) tuples.
            
        Returns:
            dict: Batch of padded features and texts.
        """
        # Separate features and texts
        features, texts = zip(*batch)
        
        # Get sequence lengths
        feat_lengths = [feat.shape[0] for feat in features]
        max_feat_len = max(feat_lengths)
        
        # Pad features
        feature_dim = features[0].shape[1]
        padded_features = np.zeros((len(features), max_feat_len, feature_dim))
        
        for i, feat in enumerate(features):
            padded_features[i, :feat.shape[0], :] = feat
        
        # TODO: Process and pad texts
        # For now, just return as is
        
        return {
            'features': paddle.to_tensor(padded_features, dtype='float32'),
            'feature_lengths': paddle.to_tensor(feat_lengths, dtype='int64'),
            'texts': texts
        }
