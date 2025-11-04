"""
Data preprocessing utilities for English-Darija translation
"""
import pandas as pd
from sklearn.model_selection import train_test_split
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Handle all data preprocessing tasks"""
    
    def __init__(self, data_path: str):
        """
        Initialize preprocessor with data path
        
        Args:
            data_path: Path to the raw CSV dataset
        """
        self.data_path = data_path
        self.df = None
        
    def load_data(self):
        """Load dataset from CSV file"""
        logger.info(f"Loading data from {self.data_path}")
        self.df = pd.read_csv(self.data_path)
        logger.info(f"Loaded {len(self.df)} rows")
        return self
    
    def clean_data(self):
        """Clean the dataset by handling missing values"""
        logger.info("Cleaning data...")
        
        # Check for missing values
        missing_before = self.df.isnull().sum()
        logger.info(f"Missing values before cleaning:\n{missing_before}")
        
        # Drop rows with missing values
        self.df.dropna(inplace=True)
        
        missing_after = self.df.isnull().sum()
        logger.info(f"Missing values after cleaning:\n{missing_after}")
        logger.info(f"Remaining rows: {len(self.df)}")
        
        return self
    
    def remove_duplicates(self):
        """Remove duplicate entries"""
        before = len(self.df)
        self.df.drop_duplicates(subset=['english', 'darija'], inplace=True)
        after = len(self.df)
        logger.info(f"Removed {before - after} duplicate rows")
        return self
    
    def normalize_text(self):
        """Normalize text in both columns"""
        logger.info("Normalizing text...")
        
        # Strip whitespace
        self.df['english'] = self.df['english'].str.strip()
        self.df['darija'] = self.df['darija'].str.strip()
        
        # Remove extra spaces
        self.df['english'] = self.df['english'].str.replace(r'\s+', ' ', regex=True)
        self.df['darija'] = self.df['darija'].str.replace(r'\s+', ' ', regex=True)
        
        logger.info("Text normalization complete")
        return self
    
    def split_data(self, test_size=0.2, val_size=0.5, random_state=42):
        """
        Split data into train, validation, and test sets
        
        Args:
            test_size: Proportion of data for test+validation
            val_size: Proportion of test data for validation
            random_state: Random seed for reproducibility
            
        Returns:
            tuple: (train_df, val_df, test_df)
        """
        logger.info("Splitting data...")
        
        # First split: train and temp (val+test)
        train_df, temp_df = train_test_split(
            self.df, 
            test_size=test_size, 
            random_state=random_state
        )
        
        # Second split: val and test
        val_df, test_df = train_test_split(
            temp_df, 
            test_size=val_size, 
            random_state=random_state
        )
        
        logger.info(f"Training set size: {len(train_df)}")
        logger.info(f"Validation set size: {len(val_df)}")
        logger.info(f"Test set size: {len(test_df)}")
        
        return train_df, val_df, test_df
    
    def save_splits(self, train_df, val_df, test_df, output_dir):
        """
        Save the split datasets to CSV files
        
        Args:
            train_df: Training dataframe
            val_df: Validation dataframe
            test_df: Test dataframe
            output_dir: Directory to save the files
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        train_path = os.path.join(output_dir, 'train.csv')
        val_path = os.path.join(output_dir, 'val.csv')
        test_path = os.path.join(output_dir, 'test.csv')
        
        train_df.to_csv(train_path, index=False)
        val_df.to_csv(val_path, index=False)
        test_df.to_csv(test_path, index=False)
        
        logger.info(f"Saved train data to {train_path}")
        logger.info(f"Saved validation data to {val_path}")
        logger.info(f"Saved test data to {test_path}")
    
    def process_pipeline(self, output_dir='../data/processed'):
        """
        Run the complete preprocessing pipeline
        
        Args:
            output_dir: Directory to save processed data
            
        Returns:
            tuple: (train_df, val_df, test_df)
        """
        self.load_data()
        self.clean_data()
        self.remove_duplicates()
        self.normalize_text()
        
        train_df, val_df, test_df = self.split_data()
        self.save_splits(train_df, val_df, test_df, output_dir)
        
        logger.info("Preprocessing pipeline complete!")
        return train_df, val_df, test_df


def main():
    """Main execution function"""
    # Example usage
    preprocessor = DataPreprocessor(r'../data/raw/darija_english.csv')
    train_df, val_df, test_df = preprocessor.process_pipeline()
    
    print("\n=== Preprocessing Complete ===")
    print(f"Train samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")
    print(f"Test samples: {len(test_df)}")


if __name__ == "__main__":
    main()