"""
Training script for English-Darija translation model
"""
import os
import argparse
import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TranslationTrainer:
    """Handle model training for translation task"""
    
    def __init__(self, model_name="facebook/nllb-200-distilled-600M", 
                 src_lang="eng_Latn", tgt_lang="ary_Arab"):
        """
        Initialize trainer with model and language settings
        
        Args:
            model_name: Pretrained model identifier
            src_lang: Source language code
            tgt_lang: Target language code
        """
        self.model_name = model_name
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        
        logger.info(f"Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        
        # Set language codes
        self.tokenizer.src_lang = src_lang
        self.tokenizer.tgt_lang = tgt_lang
        
        logger.info("Model and tokenizer loaded successfully")
    
    def load_datasets(self, train_path, val_path):
        """
        Load training and validation datasets
        
        Args:
            train_path: Path to training CSV
            val_path: Path to validation CSV
            
        Returns:
            tuple: (train_dataset, val_dataset)
        """
        logger.info("Loading datasets...")
        
        train_df = pd.read_csv(train_path)
        val_df = pd.read_csv(val_path)
        
        logger.info(f"Train samples: {len(train_df)}")
        logger.info(f"Validation samples: {len(val_df)}")
        
        return train_df, val_df
    
    def tokenize_function(self, examples):
        """
        Tokenize examples for seq2seq training
        
        Args:
            examples: Batch of examples
            
        Returns:
            dict: Tokenized inputs and labels
        """
        # Tokenize English sentences as inputs
        self.tokenizer.src_lang = self.src_lang
        inputs = self.tokenizer(examples["english"], truncation=True, max_length=128)
        
        # Tokenize Darija sentences as labels
        self.tokenizer.tgt_lang = self.tgt_lang
        labels = self.tokenizer(examples["darija"], truncation=True, max_length=128)
        
        # Add labels to inputs
        inputs["labels"] = labels["input_ids"]
        return inputs
    
    def prepare_datasets(self, train_df, val_df):
        """
        Prepare datasets for training
        
        Args:
            train_df: Training dataframe
            val_df: Validation dataframe
            
        Returns:
            tuple: (train_dataset, val_dataset)
        """
        logger.info("Tokenizing datasets...")
        
        # Convert to HuggingFace Dataset
        train_dataset = Dataset.from_pandas(train_df)
        val_dataset = Dataset.from_pandas(val_df)
        
        # Apply tokenization
        train_dataset = train_dataset.map(
            self.tokenize_function, 
            batched=True, 
            remove_columns=["english", "darija"]
        )
        val_dataset = val_dataset.map(
            self.tokenize_function, 
            batched=True, 
            remove_columns=["english", "darija"]
        )
        
        logger.info("Tokenization complete")
        return train_dataset, val_dataset
    
    def train(self, train_dataset, val_dataset, output_dir, 
              num_epochs=5, batch_size=8, learning_rate=2e-5):
        """
        Train the model
        
        Args:
            train_dataset: Tokenized training dataset
            val_dataset: Tokenized validation dataset
            output_dir: Directory to save model checkpoints
            num_epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate
        """
        logger.info("Setting up training...")
        
        # Define training arguments
        training_args = Seq2SeqTrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=0.01,
            eval_strategy="steps",
            eval_steps=500,
            save_steps=1000,
            save_total_limit=3,
            predict_with_generate=True,
            logging_dir=f"{output_dir}/logs",
            logging_steps=100,
            load_best_model_at_end=True,
            metric_for_best_model="loss",
            greater_is_better=False,
            fp16=torch.cuda.is_available(),  # Use mixed precision if GPU available
            report_to=["tensorboard"],
        )
        
        # Data collator
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer, 
            model=self.model
        )
        
        # Initialize trainer
        trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
        )
        
        # Start training
        logger.info("Starting training...")
        trainer.train()
        
        # Save final model
        logger.info(f"Saving final model to {output_dir}")
        trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        logger.info("Training complete!")
        return trainer


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train English-Darija translation model")
    
    parser.add_argument(
        "--train_data",
        type=str,
        default="../data/processed/train.csv",
        help="Path to training data CSV"
    )
    parser.add_argument(
        "--val_data",
        type=str,
        default="../data/processed/val.csv",
        help="Path to validation data CSV"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="../models/nllb-fine-tuned",
        help="Directory to save model checkpoints"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="facebook/nllb-200-distilled-600M",
        help="Pretrained model name"
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=5,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for training"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-5,
        help="Learning rate"
    )
    
    return parser.parse_args()


def main():
    """Main execution function"""
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize trainer
    trainer = TranslationTrainer(model_name=args.model_name)
    
    # Load datasets
    train_df, val_df = trainer.load_datasets(args.train_data, args.val_data)
    
    # Prepare datasets
    train_dataset, val_dataset = trainer.prepare_datasets(train_df, val_df)
    
    # Train model
    trainer.train(
        train_dataset,
        val_dataset,
        output_dir=args.output_dir,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )
    
    print("\n=== Training Complete ===")
    print(f"Model saved to: {args.output_dir}")


if __name__ == "__main__":
    main()