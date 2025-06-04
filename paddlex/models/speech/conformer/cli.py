"""
Command-line interface for Conformer model training and inference in PaddleX.

This module provides a command-line interface for training and running inference
with Conformer models for speech recognition tasks.
"""

import os
import sys
import argparse
import logging
import paddle

from .model import ConformerModel
from .trainer import ConformerTrainer
from .predictor import ConformerPredictor
from .config import ConformerConfig
from .data import AudioDataset, AudioCollator


def setup_logger(log_level="INFO"):
    """Set up logger configuration."""
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")
    
    logging.basicConfig(
        format='%(asctime)s [%(levelname)s] %(message)s',
        level=numeric_level
    )


def train(args):
    """
    Run training for Conformer model.
    
    Args:
        args (argparse.Namespace): Command-line arguments.
    """
    # Set device
    paddle.set_device(args.device)
    
    # Load configuration
    config = ConformerConfig(args.config_file)
    
    # Override config with command-line arguments
    config._update_config(vars(args))
    
    # Log configuration
    logging.info(f"Configuration:\n{config}")
    
    # Create save directory if it doesn't exist
    save_dir = config.training_config['save_dir']
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Save the configuration
    config.save(os.path.join(save_dir, "config.yaml"))
    
    # Create datasets
    logging.info("Creating datasets...")
    train_dataset = AudioDataset(
        args.train_manifest,
        config.data_config,
        mode="train"
    )
    
    dev_dataset = None
    if args.dev_manifest:
        dev_dataset = AudioDataset(
            args.dev_manifest,
            config.data_config,
            mode="dev"
        )
    
    # Create data loaders
    collator = AudioCollator()
    train_loader = paddle.io.DataLoader(
        train_dataset,
        batch_size=config.training_config['batch_size'],
        shuffle=True,
        collate_fn=collator,
        num_workers=args.num_workers
    )
    
    dev_loader = None
    if dev_dataset:
        dev_loader = paddle.io.DataLoader(
            dev_dataset,
            batch_size=config.training_config['batch_size'],
            shuffle=False,
            collate_fn=collator,
            num_workers=args.num_workers
        )
    
    # Create model
    logging.info("Creating model...")
    model = ConformerModel(**config.model_config)
    
    # Create optimizer
    logging.info("Setting up optimizer...")
    optimizer = paddle.optimizer.Adam(
        parameters=model.parameters(),
        learning_rate=config.training_config['learning_rate'],
        weight_decay=config.training_config['weight_decay']
    )
    
    # Create criterion
    # TODO: Create appropriate criterion based on task (e.g., CTC loss)
    
    # Create trainer
    trainer = ConformerTrainer(
        model=model,
        optimizer=optimizer,
        criterion=None,  # TODO: Add proper criterion
        config=config.training_config,
        train_loader=train_loader,
        dev_loader=dev_loader
    )
    
    # Start training
    logging.info("Starting training...")
    trainer.train()


def infer(args):
    """
    Run inference with Conformer model.
    
    Args:
        args (argparse.Namespace): Command-line arguments.
    """
    # Set device
    paddle.set_device(args.device)
    
    # Create predictor
    predictor = ConformerPredictor(args.model_path)
    
    # Run inference
    results = predictor.predict(args.audio_file)
    
    # Print results
    print("\nInference Results:")
    print(f"Transcription: {results['results']}")
    print(f"Processing time: {results['elapsed_time']:.2f} seconds")


def main():
    """
    Main entry point for the command-line interface.
    """
    parser = argparse.ArgumentParser(description="PaddleX Conformer Model CLI")
    subparsers = parser.add_subparsers(dest="mode", help="Mode of operation")
    
    # Common arguments
    common_parser = argparse.ArgumentParser(add_help=False)
    common_parser.add_argument(
        "--device", 
        default="gpu", 
        choices=["cpu", "gpu"],
        help="Device to run on"
    )
    common_parser.add_argument(
        "--log-level", 
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    
    # Train mode parser
    train_parser = subparsers.add_parser(
        "train", 
        parents=[common_parser],
        help="Train a Conformer model"
    )
    train_parser.add_argument(
        "--train-manifest", 
        required=True,
        help="Path to training manifest file"
    )
    train_parser.add_argument(
        "--dev-manifest",
        help="Path to validation manifest file"
    )
    train_parser.add_argument(
        "--config-file",
        help="Path to configuration file"
    )
    train_parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of workers for data loading"
    )
    train_parser.add_argument(
        "--checkpoint",
        help="Path to checkpoint to resume training from"
    )
    
    # Infer mode parser
    infer_parser = subparsers.add_parser(
        "infer", 
        parents=[common_parser],
        help="Run inference with a trained Conformer model"
    )
    infer_parser.add_argument(
        "--model-path", 
        required=True,
        help="Path to model checkpoint"
    )
    infer_parser.add_argument(
        "--audio-file", 
        required=True,
        help="Path to audio file for inference"
    )
    infer_parser.add_argument(
        "--config-file",
        help="Path to configuration file"
    )
    
    args = parser.parse_args()
    
    # Set up logger
    setup_logger(args.log_level)
    
    # Run appropriate mode
    if args.mode == "train":
        train(args)
    elif args.mode == "infer":
        infer(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
