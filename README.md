# Dendritic Cyber Sentinel

A network anomaly detection system based on the Multi-Resolution Analysis Deterministic Dendritic Cell Algorithm with Segmentation (MRA S-dDCA).

## Overview

Dendritic Cyber Sentinel implements the MRA S-dDCA algorithm for network anomaly detection, as described in the paper "Multiresolution dendritic cell algorithm for network anomaly detection" by David Limon-Cantu and Vicente Alarcon-Aquino (2021).

The algorithm is inspired by the behavior of dendritic cells in the human immune system and combines:

1. **Feature selection** using mutual information
2. **Multiresolution analysis** using wavelet transforms
3. **Dendritic Cell Algorithm (DCA)** for anomaly detection
4. **Segmentation approach** for real-time analysis

This implementation is faithful to the original MATLAB code and paper, with improvements for better usability and integration with Python data science tools.

## Installation

### Requirements

- Python 3.8 or higher
- NumPy
- Pandas
- scikit-learn
- PyWavelets

### Install from source

```bash
git clone https://github.com/example/dendritic-cyber-sentinel.git
cd dendritic-cyber-sentinel
pip install -e .
```

## Usage

### Basic Example

```python
from dendritic_cyber_sentinel import MRA_SdDCA, KDDLoader, prepare_kdd_dataset

# Load dataset
loader = KDDLoader()
dataset = loader.load_data("path/to/KDDTest+.csv")

# Prepare dataset
prepared_dataset = prepare_kdd_dataset(dataset)

# Initialize and run MRA S-dDCA
mra_sdca = MRA_SdDCA(verbose=True)
confmat, accuracy, runtime = mra_sdca.fit_predict(
    data=prepared_dataset,
    name="NSL-KDD",
    T=5,           # Number of features per signal category
    S=2204,        # Random seed
    m=128,         # Segment size
    p=1,           # Dendritic cell population size
    wavelet="db1"  # Daubechies 1 wavelet
)

# Print results
print(f"Confusion Matrix [TP, TN, FP, FN]: {confmat}")
print(f"Accuracy: {accuracy:.4f}")
print(f"Runtime: {runtime:.4f} seconds")
```

### Command Line Interface

The package provides command-line tools for running the algorithm on standard datasets:

```bash
# Run on NSL-KDD dataset
dcs-nsl-kdd --dataset path/to/KDDTest+.csv --output results/ --verbose

# Run on UNSW-NB15 dataset
dcs-unsw-nb15 --dataset path/to/UNSW-NB15.csv --output results/ --verbose
```

## Supported Datasets

1. **NSL-KDD** - An improved version of the KDD Cup 1999 dataset for network intrusion detection
2. **UNSW-NB15** - A contemporary network traffic dataset with modern attack types

## Algorithm Parameters

- **T**: Number of features to select for each signal category
- **S**: Random seed for reproducibility
- **m**: Segment size for processing data in chunks
- **p**: Dendritic cell population size
- **wavelet**: Wavelet type for multiresolution analysis (e.g., "db1", "sym2")

## References

1. Limon-Cantu, D., & Alarcon-Aquino, V. (2021). Multiresolution dendritic cell algorithm for network anomaly detection. PeerJ Computer Science, 7, e749.
2. Greensmith, J., & Aickelin, U. (2008). The Deterministic Dendritic Cell Algorithm. In Artificial Immune Systems (pp. 291-302).
3. Gu, F., Greensmith, J., & Aickelin, U. (2009). Integrating Real-Time Analysis with the Dendritic Cell Algorithm through Segmentation. In Proceedings of the 11th Annual Conference on Genetic and Evolutionary Computation (pp. 1203-1210).

## License

This project is licensed under the MIT License - see the LICENSE file for details.
