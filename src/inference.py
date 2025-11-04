"""
Inference utilities for English-Darija translation
"""
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Translator:
    """Handle translation inference"""
    
    def __init__(self, model_path, device=None):
        """
        Initialize translator with trained model
        
        Args:
            model_path: Path to the fine-tuned model
            device: Device to run inference on ('cuda' or 'cpu')
        """
        logger.info(f"Loading model from {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        
        # Set device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        self.model.to(self.device)
        logger.info(f"Model loaded on device: {self.device}")
    
    def translate_english_to_darija(self, text, max_length=128, num_beams=5):
        """
        Translate English text to Darija
        
        Args:
            text: English text to translate
            max_length: Maximum length of translation
            num_beams: Number of beams for beam search
            
        Returns:
            str: Translated Darija text
        """
        # Set source and target languages
        self.tokenizer.src_lang = "eng_Latn"
        
        # Tokenize input
        inputs = self.tokenizer(text, return_tensors="pt", padding=True).to(self.device)
        
        # Generate translation
        with torch.no_grad():
            translated_tokens = self.model.generate(
                **inputs,
                forced_bos_token_id=self.tokenizer.convert_tokens_to_ids("__ary_Arab__"),
                max_length=max_length,
                num_beams=num_beams,
                early_stopping=True
            )
        
        # Decode translation
        translation = self.tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
        return translation
    
    def translate_darija_to_english(self, text, max_length=128, num_beams=5):
        """
        Translate Darija text to English
        
        Args:
            text: Darija text to translate
            max_length: Maximum length of translation
            num_beams: Number of beams for beam search
            
        Returns:
            str: Translated English text
        """
        # Set source and target languages
        self.tokenizer.src_lang = "ary_Arab"
        
        # Tokenize input
        inputs = self.tokenizer(text, return_tensors="pt", padding=True).to(self.device)
        
        # Generate translation
        with torch.no_grad():
            translated_tokens = self.model.generate(
                **inputs,
                forced_bos_token_id=self.tokenizer.convert_tokens_to_ids("__eng_Latn__"),
                max_length=max_length,
                num_beams=num_beams,
                early_stopping=True
            )
        
        # Decode translation
        translation = self.tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
        return translation
    
    def translate_batch(self, texts, direction="en2dar", max_length=128, batch_size=8):
        """
        Translate a batch of texts
        
        Args:
            texts: List of texts to translate
            direction: Translation direction ('en2dar' or 'dar2en')
            max_length: Maximum length of translations
            batch_size: Batch size for processing
            
        Returns:
            list: List of translations
        """
        translations = []
        
        # Set language codes based on direction
        if direction == "en2dar":
            self.tokenizer.src_lang = "eng_Latn"
            tgt_lang_id = self.tokenizer.convert_tokens_to_ids("__ary_Arab__")
        else:
            self.tokenizer.src_lang = "ary_Arab"
            tgt_lang_id = self.tokenizer.convert_tokens_to_ids("__eng_Latn__")
        
        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            # Tokenize batch
            inputs = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(self.device)
            
            # Generate translations
            with torch.no_grad():
                translated_tokens = self.model.generate(
                    **inputs,
                    forced_bos_token_id=tgt_lang_id,
                    max_length=max_length,
                    num_beams=5,
                    early_stopping=True
                )
            
            # Decode translations
            batch_translations = self.tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)
            translations.extend(batch_translations)
        
        return translations
    
    def interactive_mode(self):
        """Run translator in interactive mode"""
        print("\n=== English ↔ Darija Translator ===")
        print("Commands:")
        print("  - Type English text to translate to Darija")
        print("  - Type ':dar' followed by Darija text to translate to English")
        print("  - Type ':quit' to exit\n")
        
        while True:
            try:
                user_input = input("You: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() == ":quit":
                    print("Goodbye!")
                    break
                
                if user_input.startswith(":dar "):
                    # Translate Darija to English
                    darija_text = user_input[5:]
                    translation = self.translate_darija_to_english(darija_text)
                    print(f"English: {translation}\n")
                else:
                    # Translate English to Darija
                    translation = self.translate_english_to_darija(user_input)
                    print(f"Darija: {translation}\n")
            
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}\n")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Translate between English and Darija")
    
    parser.add_argument(
        "--model_path",
        type=str,
        default="../models/nllb-fine-tuned",
        help="Path to the fine-tuned model"
    )
    parser.add_argument(
        "--text",
        type=str,
        help="Text to translate"
    )
    parser.add_argument(
        "--direction",
        type=str,
        default="en2dar",
        choices=["en2dar", "dar2en"],
        help="Translation direction (en2dar or dar2en)"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode"
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "cpu"],
        help="Device to run inference on"
    )
    
    return parser.parse_args()


def main():
    """Main execution function"""
    args = parse_args()
    
    # Initialize translator
    translator = Translator(model_path=args.model_path, device=args.device)
    
    if args.interactive:
        # Run in interactive mode
        translator.interactive_mode()
    elif args.text:
        # Translate single text
        if args.direction == "en2dar":
            translation = translator.translate_english_to_darija(args.text)
            print(f"\nEnglish: {args.text}")
            print(f"Darija: {translation}\n")
        else:
            translation = translator.translate_darija_to_english(args.text)
            print(f"\nDarija: {args.text}")
            print(f"English: {translation}\n")
    else:
        print("Please provide --text or use --interactive mode")
        print("Example: python inference.py --text 'Hello, how are you?' --direction en2dar")
        print("Or: python inference.py --interactive")


if __name__ == "__main__":
    main()