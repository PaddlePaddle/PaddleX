"""
Configuration module for Conformer models in PaddleX.

This module provides utilities for managing configuration of Conformer models,
training parameters, and data processing settings.
"""

import os
import yaml


class ConformerConfig:
    """
    Configuration class for Conformer model.
    
    Handles loading and saving of configuration parameters.
    
    Args:
        config_path (str, optional): Path to a YAML configuration file. Defaults to None.
        **kwargs: Additional configuration parameters to override defaults.
    """
    
    def __init__(self, config_path=None, **kwargs):
        # Default configurations
        self.model_config = {
            "input_dim": 80,
            "output_dim": 4096,  # Vocabulary size
            "encoder_dim": 256,
            "num_encoder_layers": 12,
            "num_attention_heads": 4,
            "feed_forward_expansion_factor": 4,
            "conv_expansion_factor": 2,
            "feed_forward_dropout_p": 0.1,
            "attention_dropout_p": 0.1,
            "conv_dropout_p": 0.1,
            "conv_kernel_size": 31,
        }
        
        self.training_config = {
            "epochs": 100,
            "batch_size": 16,
            "learning_rate": 1e-3,
            "weight_decay": 1e-6,
            "lr_scheduler": "cosine",
            "warmup_steps": 10000,
            "log_interval": 100,
            "save_interval": 5,
            "save_dir": "./checkpoints",
        }
        
        self.data_config = {
            "feature_type": "fbank",
            "sample_rate": 16000,
            "n_fft": 512,
            "win_length": 400,
            "hop_length": 160,
            "n_mels": 80,
            "normalization": "global",
            "spec_augment": True,
            "spec_aug_config": {
                "time_warp": False,
                "freq_mask": 27,
                "time_mask": 100,
                "n_freq_mask": 2,
                "n_time_mask": 2,
            },
        }
        
        # Load configuration from file if provided
        if config_path and os.path.isfile(config_path):
            self._load_from_file(config_path)
        
        # Override configurations with any provided kwargs
        self._update_config(kwargs)
    
    def _load_from_file(self, config_path):
        """
        Load configuration from YAML file.
        
        Args:
            config_path (str): Path to the YAML configuration file.
        """
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        if 'model_config' in config:
            self.model_config.update(config['model_config'])
        
        if 'training_config' in config:
            self.training_config.update(config['training_config'])
        
        if 'data_config' in config:
            self.data_config.update(config['data_config'])
    
    def _update_config(self, params):
        """
        Update configuration with provided parameters.
        
        Args:
            params (dict): Dictionary of parameters to update.
        """
        for key, value in params.items():
            if key.startswith('model_'):
                _key = key[6:]  # Remove 'model_' prefix
                if _key in self.model_config:
                    self.model_config[_key] = value
            elif key.startswith('train_'):
                _key = key[6:]  # Remove 'train_' prefix
                if _key in self.training_config:
                    self.training_config[_key] = value
            elif key.startswith('data_'):
                _key = key[5:]  # Remove 'data_' prefix
                if _key in self.data_config:
                    self.data_config[_key] = value
    
    def save(self, config_path):
        """
        Save configuration to a YAML file.
        
        Args:
            config_path (str): Path to save the configuration file.
        """
        config = {
            "model_config": self.model_config,
            "training_config": self.training_config,
            "data_config": self.data_config,
        }
        
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
    
    def __str__(self):
        """String representation of the configuration."""
        config = {
            "model_config": self.model_config,
            "training_config": self.training_config,
            "data_config": self.data_config,
        }
        return yaml.dump(config)
