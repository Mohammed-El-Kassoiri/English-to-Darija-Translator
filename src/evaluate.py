"""
Evaluation script for English-Darija translation model
"""
import argparse
import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq
from evaluate import load
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TranslationEvaluator:
    """Handle model evaluation for translation task"""
    
    def __init__(self, model_path, src_lang="eng_Latn", tgt_lang="ary_Arab"):
        """
        Initialize evaluator with trained model
        
        Args:
            model_path: Path to the fine-tuned model
            src_lang: Source language code
            tgt_lang: Target language code
        """
        logger.info(f"Loading model from {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        
        # Move model to GPU if available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        logger.info(f"Model loaded on device: {self.device}")
        
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.tokenizer.src_lang = src_lang
        self.tokenizer.tgt_lang = tgt_lang
        
        # Load metrics
        logger.info("Loading evaluation metrics...")
        self.bleu_metric = load("bleu")
        self.meteor_metric = load("meteor")
        self.chrf_metric = load("chrf")
    
    def load_test_data(self, test_path):
        """
        Load test dataset
        
        Args:
            test_path: Path to test CSV file
            
        Returns:
            DataFrame: Test dataframe
        """
        logger.info(f"Loading test data from {test_path}")
        test_df = pd.read_csv(test_path)
        logger.info(f"Loaded {len(test_df)} test samples")
        return test_df
    
    def translate(self, text, max_length=128):
        """
        Translate a single text
        
        Args:
            text: Input text in source language
            max_length: Maximum length of translation
            
        Returns:
            str: Translated text
        """
        self.tokenizer.src_lang = self.src_lang
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        
        translated_tokens = self.model.generate(
            **inputs,
            forced_bos_token_id=self.tokenizer.convert_tokens_to_ids(f"__{self.tgt_lang}__"),
            max_length=max_length
        )
        
        translation = self.tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
        return translation
    
    def evaluate_on_test_set(self, test_df):
        """
        Evaluate model on full test set
        
        Args:
            test_df: Test dataframe with 'english' and 'darija' columns
            
        Returns:
            dict: Evaluation metrics
        """
        logger.info("Starting evaluation on test set...")
        
        predictions = []
        references = []
        
        for idx, row in test_df.iterrows():
            if (idx + 1) % 100 == 0:
                logger.info(f"Processed {idx + 1}/{len(test_df)} samples")
            
            english_text = row['english']
            darija_reference = row['darija']
            
            # Generate translation
            prediction = self.translate(english_text)
            
            predictions.append(prediction)
            references.append(darija_reference)
        
        logger.info("Calculating metrics...")
        
        # Calculate BLEU
        bleu_results = self.bleu_metric.compute(
            predictions=predictions,
            references=[[ref] for ref in references]
        )
        
        # Calculate METEOR
        meteor_results = self.meteor_metric.compute(
            predictions=predictions,
            references=references
        )
        
        # Calculate chrF
        chrf_results = self.chrf_metric.compute(
            predictions=predictions,
            references=[[ref] for ref in references]
        )
        
        metrics = {
            "bleu": bleu_results["bleu"],
            "meteor": meteor_results["meteor"],
            "chrf": chrf_results["score"]
        }
        
        return metrics, predictions, references
    
    def evaluate_with_trainer(self, test_path):
        """
        Alternative evaluation using Seq2SeqTrainer
        
        Args:
            test_path: Path to test CSV
            
        Returns:
            dict: Evaluation results
        """
        logger.info("Evaluating with Seq2SeqTrainer...")
        
        # Load test data
        test_df = pd.read_csv(test_path)
        
        # Tokenize
        def tokenize_function(examples):
            self.tokenizer.src_lang = self.src_lang
            inputs = self.tokenizer(examples["english"], truncation=True, max_length=128)
            
            self.tokenizer.tgt_lang = self.tgt_lang
            labels = self.tokenizer(examples["darija"], truncation=True, max_length=128)
            
            inputs["labels"] = labels["input_ids"]
            return inputs
        
        test_dataset = Dataset.from_pandas(test_df)
        test_dataset = test_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=["english", "darija"]
        )
        
        # Setup trainer for evaluation
        training_args = Seq2SeqTrainingArguments(
            output_dir="./eval_results",
            per_device_eval_batch_size=8,
            predict_with_generate=True,
        )
        
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=self.model
        )
        
        trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
        )
        
        # Evaluate
        results = trainer.evaluate(test_dataset)
        
        logger.info("Evaluation complete!")
        return results
    
    def show_examples(self, test_df, num_examples=5):
        """
        Show example translations
        
        Args:
            test_df: Test dataframe
            num_examples: Number of examples to show
        """
        logger.info(f"\n=== Showing {num_examples} Example Translations ===\n")
        
        for idx in range(min(num_examples, len(test_df))):
            row = test_df.iloc[idx]
            english_text = row['english']
            darija_reference = row['darija']
            prediction = self.translate(english_text)
            
            print(f"Example {idx + 1}:")
            print(f"  English: {english_text}")
            print(f"  Reference Darija: {darija_reference}")
            print(f"  Predicted Darija: {prediction}")
            print()


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Evaluate English-Darija translation model")
    
    parser.add_argument(
        "--model_path",
        type=str,
        default="../models/nllb-fine-tuned",
        help="Path to the fine-tuned model"
    )
    parser.add_argument(
        "--test_data",
        type=str,
        default="../data/processed/test.csv",
        help="Path to test data CSV"
    )
    parser.add_argument(
        "--num_examples",
        type=int,
        default=5,
        help="Number of example translations to show"
    )
    parser.add_argument(
        "--use_trainer",
        action="store_true",
        help="Use Seq2SeqTrainer for evaluation"
    )
    
    return parser.parse_args()


def main():
    """Main execution function"""
    args = parse_args()
    
    # Initialize evaluator
    evaluator = TranslationEvaluator(model_path=args.model_path)
    
    # Load test data
    test_df = evaluator.load_test_data(args.test_data)
    
    # Show example translations
    evaluator.show_examples(test_df, num_examples=args.num_examples)
    
    if args.use_trainer:
        # Evaluate using trainer
        results = evaluator.evaluate_with_trainer(args.test_data)
        print("\n=== Evaluation Results (Trainer) ===")
        for key, value in results.items():
            print(f"{key}: {value:.4f}")
    else:
        # Evaluate on test set
        metrics, predictions, references = evaluator.evaluate_on_test_set(test_df)
        
        print("\n=== Evaluation Metrics ===")
        print(f"BLEU Score: {metrics['bleu']:.4f}")
        print(f"METEOR Score: {metrics['meteor']:.4f}")
        print(f"chrF Score: {metrics['chrf']:.4f}")
        
        # Save predictions
        results_df = test_df.copy()
        results_df['prediction'] = predictions
        results_path = "../data/processed/test_predictions.csv"
        results_df.to_csv(results_path, index=False)
        print(f"\nPredictions saved to: {results_path}")


if __name__ == "__main__":
    main()