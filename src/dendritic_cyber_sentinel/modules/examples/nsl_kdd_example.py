"""
Example script for NSL-KDD dataset using MRA S-dDCA.

This script demonstrates how to use the MRA S-dDCA algorithm with the NSL-KDD dataset.
"""

import os
import sys
import argparse
from pathlib import Path

from ..mra_sdca import MRA_SdDCA
from ..data_loaders import KDDLoader, prepare_kdd_dataset


def main():
    """Main function to demonstrate MRA S-dDCA on NSL-KDD dataset."""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run MRA S-dDCA on NSL-KDD dataset')
    parser.add_argument('--dataset', type=str, default="KDDTest+.csv",
                        help='Path to the KDD dataset CSV file')
    parser.add_argument('--output', type=str, default=None,
                        help='Directory to save results (default: current directory)')
    parser.add_argument('--features', type=int, default=5,
                        help='Number of features to select for each signal category (default: 5)')
    parser.add_argument('--seed', type=int, default=2204,
                        help='Random seed (default: 2204)')
    parser.add_argument('--segment', type=int, default=128,
                        help='Segment size (default: 128)')
    parser.add_argument('--population', type=int, default=1,
                        help='Dendritic cell population size (default: 1)')
    parser.add_argument('--wavelet', type=str, default="db1",
                        help='Wavelet to use (default: db1)')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose output')
    
    args = parser.parse_args()
    
    print("MRA S-dDCA Example: NSL-KDD Dataset")
    print("===================================")
    
    # Initialize data loader
    loader = KDDLoader()
    
    # Load the dataset
    print(f"Loading dataset from: {args.dataset}")
    try:
        dataset = loader.load_data(args.dataset)
        print(f"Loaded dataset with {len(dataset)} samples")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Please ensure the KDD dataset file is available and the path is correct.")
        return 1
    
    # Prepare dataset for MRA S-dDCA
    prepared_dataset = prepare_kdd_dataset(dataset)
    print(f"Prepared dataset with {len(prepared_dataset)} samples and {len(prepared_dataset.columns)} features")
    
    # Initialize MRA S-dDCA algorithm
    mra_sdca = MRA_SdDCA(verbose=args.verbose)
    
    # Run MRA S-dDCA algorithm
    print("\nRunning MRA S-dDCA algorithm...")
    try:
        results = mra_sdca.fit_predict(
            data=prepared_dataset,
            name="NSL-KDD",
            T=args.features,
            S=args.seed,
            m=args.segment,
            p=args.population,
            wavelet=args.wavelet,
            output_dir=args.output,
            verbose=args.verbose
        )
        
        # Extract results
        confmat = results['confmat']
        accuracy = results['accuracy']
        runtime = results['runtime']
        multi_class_accuracy = results.get('multi_class_accuracy')
        
        # Display results
        print(f"\n=== Results ===")
        print(f"Confusion Matrix [TP, TN, FP, FN]: {confmat}")
        print(f"Binary Accuracy: {accuracy:.4f}")
        print(f"Runtime: {runtime:.4f} seconds")
        
        # Calculate additional metrics
        tp, tn, fp, fn = confmat
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        print(f"Precision: {precision:.4f}")
        print(f"Sensitivity (Recall): {sensitivity:.4f}")
        print(f"Specificity: {specificity:.4f}")
        
        # Show multi-class results if available
        if multi_class_accuracy is not None:
            print(f"\n=== Multi-class Classification ===")
            print(f"Multi-class Accuracy: {multi_class_accuracy:.4f}")
        
        return 0
        
    except Exception as e:
        print(f"Error running MRA S-dDCA: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main()) 