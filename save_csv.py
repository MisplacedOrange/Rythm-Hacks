import pandas as pd
import numpy as np
from pathlib import Path

def save_heart_disease_to_csv(output_filename='heart_disease_data.csv'):
    """
    Load the Heart Disease dataset from UCI and save it as a CSV file.
    
    Parameters:
    -----------
    output_filename : str
        Name of the output CSV file (default: 'heart_disease_data.csv')
    """
    print("Loading Heart Disease dataset from UCI...")
    
    try:
        from ucimlrepo import fetch_ucirepo
        
        # Fetch dataset
        heart_disease = fetch_ucirepo(id=45)
        
        # Get features and targets
        X = heart_disease.data.features
        y = heart_disease.data.targets
        
        # Handle missing values in features
        X = X.fillna(X.mean())
        
        # Convert target to binary: 0 = no disease, 1+ = disease present
        y_binary = (y > 0).astype(int)
        y_binary.columns = ['target']  # Rename column
        
        # Combine features and target into one dataframe
        full_data = pd.concat([X, y_binary], axis=1)
        
        # Save to CSV
        full_data.to_csv(output_filename, index=False)
        
        print(f"\n✓ Successfully saved dataset to: {output_filename}")
        print(f"  - Total rows: {len(full_data)}")
        print(f"  - Total columns: {len(full_data.columns)}")
        print(f"  - Features: {list(X.columns)}")
        print(f"  - Target column: 'target' (0 = no disease, 1 = disease)")
        print(f"\nClass distribution:")
        print(full_data['target'].value_counts().sort_index())
        
        # Also save with original multi-class target (optional)
        full_data_multiclass = pd.concat([X, y], axis=1)
        multiclass_filename = output_filename.replace('.csv', '_multiclass.csv')
        full_data_multiclass.to_csv(multiclass_filename, index=False)
        print(f"\n✓ Also saved original multi-class version to: {multiclass_filename}")
        
        return full_data
        
    except ImportError:
        print("Error: ucimlrepo package not found!")
        print("Please install it using: pip install ucimlrepo")
        return None
    except Exception as e:
        print(f"Error loading or saving data: {e}")
        return None

def save_train_test_split_to_csv(test_size=0.2, random_state=42):
    """
    Load the dataset, split into train/test, and save as separate CSV files.
    
    Parameters:
    -----------
    test_size : float
        Proportion of data to use for testing (default: 0.2)
    random_state : int
        Random seed for reproducibility (default: 42)
    """
    print("Loading and splitting Heart Disease dataset...")
    
    try:
        from ucimlrepo import fetch_ucirepo
        from sklearn.model_selection import train_test_split
        
        # Fetch dataset
        heart_disease = fetch_ucirepo(id=45)
        
        # Get features and targets
        X = heart_disease.data.features
        y = heart_disease.data.targets
        
        # Handle missing values
        X = X.fillna(X.mean())
        
        # Convert target to binary
        y_binary = (y > 0).astype(int)
        y_binary.columns = ['target']
        
        # Combine features and target
        full_data = pd.concat([X, y_binary], axis=1)
        
        # Split the data
        train_data, test_data = train_test_split(
            full_data, 
            test_size=test_size, 
            random_state=random_state,
            stratify=full_data['target']
        )
        
        # Save train and test sets
        train_data.to_csv('heart_disease_train.csv', index=False)
        test_data.to_csv('heart_disease_test.csv', index=False)
        
        print(f"\n✓ Successfully saved split datasets:")
        print(f"  - Training set: heart_disease_train.csv ({len(train_data)} rows)")
        print(f"  - Test set: heart_disease_test.csv ({len(test_data)} rows)")
        print(f"\nTraining set class distribution:")
        print(train_data['target'].value_counts().sort_index())
        print(f"\nTest set class distribution:")
        print(test_data['target'].value_counts().sort_index())
        
        return train_data, test_data
        
    except ImportError:
        print("Error: Required packages not found!")
        print("Please install: pip install ucimlrepo scikit-learn")
        return None, None
    except Exception as e:
        print("Error loading or saving data: {e}")
        return None, None

def main():
    """
    Main function with options to save data in different formats.
    """
    print("="*60)
    print("Heart Disease Dataset to CSV Converter")
    print("="*60)
    
    print("\nOption 1: Save complete dataset")
    data = save_heart_disease_to_csv('heart_disease_data.csv')
    
    print("\n" + "="*60)
    print("\nOption 2: Save train/test split")
    train_data, test_data = save_train_test_split_to_csv(test_size=0.2, random_state=42)
    
    print("\n" + "="*60)
    print("\n✓ All files have been created successfully!")
    print("\nFiles created:")
    print("  1. heart_disease_data.csv - Complete dataset (binary target)")
    print("  2. heart_disease_data_multiclass.csv - Complete dataset (original multi-class target)")
    print("  3. heart_disease_train.csv - Training set (80%)")
    print("  4. heart_disease_test.csv - Test set (20%)")

if __name__ == '__main__':
    main()