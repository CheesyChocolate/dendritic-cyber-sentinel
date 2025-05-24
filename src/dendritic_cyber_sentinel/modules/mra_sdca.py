"""
Multi-Resolution Analysis Deterministic Dendritic Cell Algorithm with Segmentation (MRA S-dDCA)

This module implements the main MRA S-dDCA algorithm for network anomaly detection.
Based on the work done by Greensmith in 2008, adapted the segmentation concept 
proposed by Gu et al., 2009. Originally developed by David Limon-Cantu.

References:
[1] J. Greensmith and U. Aickelin, "The Deterministic Dendritic Cell Algorithm,"
    in Artificial Immune Systems, 2008, pp. 291–302.
[2] F. Gu, J. Greensmith, and U. Aickelin, 
    "Integrating Real-Time Analysis with the Dendritic Cell Algorithm through Segmentation," 
    in Proceedings of the 11th Annual Conference on Genetic and Evolutionary Computation, 
    New York, NY, USA, 2009, pp. 1203–1210. doi: 10.1145/1569901.1570063.
"""

import numpy as np
import pandas as pd
import pywt
import time
import csv
from datetime import datetime
from typing import Tuple, Dict, Any, Optional, Union, List
from sklearn.metrics import confusion_matrix, accuracy_score
from pathlib import Path

from .feature_selection import FeatureSelector
from .wavelet_utils import modwt, modwt_alternative


class MRA_SdDCA:
    """
    Multi-Resolution Analysis Deterministic Dendritic Cell Algorithm with Segmentation.
    
    This is a dDCA inspired intrusion detection model, developed as an
    Intrusion Detection System (IDS) approach.
    """
    
    def __init__(self, verbose: bool = True):
        """
        Initialize the MRA S-dDCA algorithm.
        
        Parameters:
        -----------
        verbose : bool, optional
            Display algorithm relevant information (default: True)
        """
        self.verbose = verbose
        self.feature_selector = FeatureSelector(verbose=verbose)
        
    def fit_predict(self, 
                   data: pd.DataFrame,
                   name: str,
                   T: int,
                   S: int,
                   m: int,
                   p: int,
                   wavelet: str,
                   output_dir: Optional[str] = None,
                   verbose: Optional[bool] = None) -> Dict[str, Any]:
        """
        Run the MRA S-dDCA algorithm on the provided dataset.
        
        This is a Python implementation of the MATLAB MRA_SdDCA.m function.
        
        Parameters:
        -----------
        data : pd.DataFrame
            High-level network feature dataset with required columns:
            - 'cat': attack labels/classes
            - 'newid_addon': categorical variable identifying network flow instances
            - 'dataset_label': binary labels (0=normal, 1=anomaly)
        name : str
            Dataset name, used for storing results in file
        T : int
            Number of features to be selected for each signal category
        S : int
            Random number generator seed used to assign migration threshold
        m : int
            Segment size used by the S-dDCA
        p : int
            Dendritic cell population size
        wavelet : str
            Wavelet used for the MODWT process (e.g., 'db1', 'sym2')
        output_dir : str, optional
            Directory to save results (default: current directory)
        verbose : bool, optional
            Override instance verbose setting
            
        Returns:
        --------
        Dict[str, Any]
            Dictionary containing:
            - 'confmat': confusion matrix [tp, tn, fp, fn]
            - 'accuracy': classification accuracy
            - 'runtime': algorithm execution time
            - 'multi_class_accuracy': accuracy for multi-class classification (if applicable)
            - 'multi_class_predictions': multi-class predictions (if applicable)
        """
        if verbose is not None:
            use_verbose = verbose
        else:
            use_verbose = self.verbose
            
        # Record start time
        start_time = time.time()
        
        # Set random seed
        np.random.seed(S)
        
        # Generate filename for results
        timestamp = datetime.now()
        filename = f"WAV_{name}-{timestamp.strftime('%Y-%m-%d-%H-%M-%S')}"
        
        # Set output directory
        if output_dir is not None:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            filename = str(output_path / filename)
        
        # Initialize results file
        self._initialize_results_file(filename)
        
        # Initialize m_og early to avoid NameError
        m_og = m
        
        try:
            # Feature selection and signal generation
            signal_dataset = self.feature_selector.select_features(data, T)
            
            # Initialize algorithm variables
            wavlvl = p
            n_samples = len(signal_dataset['newid'])
            
            # Get unique antigen categories
            antigencat = np.unique(signal_dataset['newid_addon'])
            
            # Initialize performance metrics
            tp = 0
            tn = 0
            fp = 0
            fn = 0
            
            # Initialize dendritic cell variables
            khtemp = np.zeros(p)
            csmtemp = np.zeros(p)
            alphacount = np.zeros((p, m))
            alpha = np.zeros(n_samples)
            kalpha = np.zeros(n_samples)
            mt = np.random.rand(p)  # Migration thresholds
            
            # Initialize energy storage
            energies_ds = np.zeros((n_samples, p))
            energies_ss = np.zeros((n_samples, p))
            
            # Initialize classification array
            classification = np.column_stack([signal_dataset['labels'], np.zeros(n_samples)])
            
            if use_verbose:
                print("dDCA start...")
                
            # Process first segment
            last = -1
            list_indices = np.zeros(m, dtype=int)
            
            # Get first segment data
            if m <= n_samples:
                signal_block = np.column_stack([
                    signal_dataset['ds_signal'][:m],
                    signal_dataset['ss_signal'][:m]
                ])
            else:
                signal_block = np.column_stack([
                    signal_dataset['ds_signal'],
                    signal_dataset['ss_signal']
                ])
                m = n_samples
            
            # MODWT of the first signal segment
            wav_signal_ds, wav_signal_ss = self._compute_modwt(signal_block, wavelet, wavlvl)
            
            # Signal energy calculation
            ds_energy = np.sum(wav_signal_ds**2, axis=1)
            ss_energy = np.sum(wav_signal_ss**2, axis=1)
            
            # Main processing loop
            for i in range(n_samples):
                if (i % 1000 == 0 or i == n_samples - 1) and use_verbose:
                    print(f"MRA S-dDCA iteration: {i+1} of {n_samples}")
                
                # Segment processing
                if (i % m != 0 and i < n_samples) or (i // m == i / m):
                    if last == -1:
                        if i % m == 0:
                            ind = m - 1  # Convert to 0-indexed
                        else:
                            ind = (i % m) - 1  # Convert to 0-indexed
                    else:
                        last += 1
                        ind = last
                        
                    # dDCA detection phase
                    if ind < wav_signal_ds.shape[1] and ind < wav_signal_ss.shape[1]:
                        si = np.column_stack([
                            wav_signal_ds[:p, ind],
                            wav_signal_ss[:p, ind]
                        ])
                        
                        # Store energies (ensure proper dimensions)
                        if len(ds_energy) > 0:
                            max_len = min(len(ds_energy), energies_ds.shape[1])
                            energies_ds[i, :max_len] = ds_energy[:max_len]
                        if len(ss_energy) > 0:
                            max_len = min(len(ss_energy), energies_ss.shape[1])
                            energies_ss[i, :max_len] = ss_energy[:max_len]
                        
                        # Migration check
                        migrated = csmtemp <= mt
                        
                        # Update k values for migrated cells
                        khtemp[migrated] += (si[migrated, 0] - 2 * si[migrated, 1])
                        alphacount[migrated, ind] += 1
                        csmtemp[migrated] += (si[migrated, 0] + si[migrated, 1])
                        
                        # Find cells that exceed migration threshold
                        kelem = csmtemp > mt
                        
                        if np.any(kelem):
                            alpha[i] += np.sum(alphacount[kelem, ind])
                            kalpha[i] += np.sum(khtemp[kelem])
                            
                        # Reset migrated cells
                        khtemp[kelem] = 0
                        alphacount[kelem, :] = 0
                        csmtemp[kelem] = 0
                        
                    # Store index in list
                    if i % m != 0:
                        list_indices[i % m - 1] = i
                    else:
                        list_indices[m - 1] = i
                
                # Segment processing finished - prepare next segment
                if (i % m == 0 and i < n_samples - 1 and last == -1):
                    if i + m < n_samples:
                        signal_block = np.column_stack([
                            signal_dataset['ds_signal'][i+1:i+m+1],
                            signal_dataset['ss_signal'][i+1:i+m+1]
                        ])
                    else:
                        # Handle last segment
                        remaining = n_samples - (i + 1)
                        signal_block = np.column_stack([
                            signal_dataset['ds_signal'][i+1:n_samples],
                            signal_dataset['ss_signal'][i+1:n_samples]
                        ])
                        
                        # Adjust parameters for last segment
                        wavlvl = int(np.floor(np.log2(remaining)))
                        if wavlvl < 1:
                            wavlvl = 1
                        
                        m = remaining
                        last = 0
                        
                        # Adjust migration thresholds
                        if p == 1:
                            mt = np.array([mt[0]])
                        else:
                            if p > wavlvl:
                                mt = mt[:wavlvl]
                            else:
                                mt = mt[:p]
                        
                        # Adjust population size if needed
                        if wavlvl < p:
                            p = wavlvl
                        
                        # Resize arrays
                        khtemp = khtemp[:p]
                        csmtemp = csmtemp[:p]
                        alphacount = alphacount[:p, :]
                    
                    # Next segment MODWT signal calculation
                    wav_signal_ds, wav_signal_ss = self._compute_modwt(signal_block, wavelet, wavlvl)
                    
                    # Calculate signal energies
                    ds_energy = np.sum(wav_signal_ds**2, axis=1)
                    ss_energy = np.sum(wav_signal_ss**2, axis=1)
                    
                    # Reset alphacount and list
                    alphacount = np.zeros((p, m))
                    list_indices = np.zeros(m, dtype=int)
            
            # Context assessment phase
            if use_verbose:
                print("dDCA context assessment phase")
                
            # Calculate MCAV (Mature Context Antigen Value)
            mcav = np.zeros(n_samples)
            
            # Calculate MCAV only where alpha > 0
            valid_indices = alpha > 0
            if np.any(valid_indices):
                mcav[valid_indices] = kalpha[valid_indices] / alpha[valid_indices]
                
            # Decision tree classification
            if use_verbose:
                print("Decision tree classification")
                
            # Prepare training data
            X_train = np.column_stack([mcav, alpha])

            # Check which labels to use for classification
            # If we have attack_type or attack_class columns, use them for multi-class classification
            if 'attack_type' in data.columns:
                # Use attack type for classification
                y_train = signal_dataset['attack_type']
                multi_class = True
                if use_verbose:
                    print("Using attack types for multi-class classification")
            elif 'attack_class' in data.columns and len(np.unique(data['attack_class'])) > 2:
                # Use attack class for classification
                y_train = signal_dataset['attack_class']
                multi_class = True
                if use_verbose:
                    print("Using attack class IDs for multi-class classification")
            else:
                # Use binary labels
                y_train = signal_dataset['labels']
                multi_class = False

            # Train decision tree with appropriate parameters
            from sklearn.tree import DecisionTreeClassifier
            if multi_class:
                # For multi-class, use more complex tree
                clf = DecisionTreeClassifier(
                    random_state=S,
                    max_depth=10,  # Allow deeper trees for multi-class
                    min_samples_split=5,  # Require more samples to split
                    min_samples_leaf=2    # Minimum samples in leaf nodes
                )
            else:
                # For binary classification, use simpler tree
                clf = DecisionTreeClassifier(random_state=S)

            clf.fit(X_train, y_train)

            # Make predictions
            y_pred = clf.predict(X_train)

            # Calculate confusion matrix and accuracy
            # For multi-class, we still use binary labels for the confusion matrix
            if multi_class:
                # Convert back to binary for confusion matrix calculation
                # If we predicted any attack class (not 'normal'), it's considered an attack
                if isinstance(y_pred[0], str):
                    binary_pred = (y_pred != 'normal').astype(int)
                else:
                    # Assume class_id 0 is normal, everything else is attack
                    binary_pred = (y_pred != 0).astype(int)
                
                # Store both binary and multi-class predictions
                classification[:, 1] = binary_pred
                multi_class_pred = y_pred
            else:
                # Standard binary classification
                classification[:, 1] = y_pred
                multi_class_pred = None

            confmat, accuracy = self._calculate_metrics(classification)

            # For multi-class, calculate additional metrics
            if multi_class:
                from sklearn.metrics import classification_report, accuracy_score
                if use_verbose:
                    print("Multi-class classification results:")
                    print(classification_report(y_train, y_pred))
                
                # Overall multi-class accuracy
                multi_class_accuracy = accuracy_score(y_train, y_pred)
                if use_verbose:
                    print(f"Multi-class accuracy: {multi_class_accuracy:.4f}")

            # Record runtime
            runtime = time.time() - start_time
            
            # Write results to file
            self._write_results(
                filename=filename,
                name=name,
                wavelet=wavelet,
                p=p,
                m=m_og,
                T=T,
                n_obs=n_samples,
                runtime=runtime,
                confmat=confmat,
                accuracy=accuracy
            )
            
            if use_verbose:
                print(f"Results saved to {filename}.csv")
                print(f"Confusion Matrix [TP, TN, FP, FN]: {confmat}")
                print(f"Accuracy: {accuracy:.4f}")
                print(f"Runtime: {runtime:.4f} seconds")
                
            return {
                'confmat': confmat,
                'accuracy': accuracy,
                'runtime': runtime,
                'multi_class_accuracy': multi_class_accuracy if multi_class else None,
                'multi_class_predictions': multi_class_pred if multi_class else None
            }
            
        except Exception as e:
            # Record runtime for error case
            runtime = time.time() - start_time
            
            # Write error results
            self._write_results(
                filename=filename,
                name=name,
                wavelet=wavelet,
                p=p,
                m=m_og,
                T=T,
                n_obs=0,
                runtime=runtime,
                confmat=None,
                accuracy=None,
                error=True
            )
            
            # Re-raise exception
            raise e
    
    def _compute_modwt(self, signal_block: np.ndarray, wavelet: str, 
                      levels: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute MODWT for both signal categories.
        
        Parameters:
        -----------
        signal_block : np.ndarray
            Signal block containing DS and SS signals
        wavelet : str
            Wavelet to use for MODWT
        levels : int
            Number of decomposition levels
            
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray]
            - wav_signal_ds: MODWT of DS signal
            - wav_signal_ss: MODWT of SS signal
        """
        # Ensure levels is at least 1
        if levels < 1:
            levels = 1
            
        # Ensure signal block has enough samples for decomposition
        if signal_block.shape[0] < 2**levels:
            # Adjust levels to match signal length
            levels = int(np.floor(np.log2(signal_block.shape[0])))
            if levels < 1:
                levels = 1
        
        # Extract DS and SS signals
        ds_signal = signal_block[:, 0]
        ss_signal = signal_block[:, 1]
        
        # Compute MODWT for DS signal using our custom implementation
        try:
            # Use our custom MODWT implementation
            coeffs_ds = modwt(ds_signal, wavelet, level=levels)
            # Reshape coefficients to match MATLAB format (levels x samples)
            wav_signal_ds = np.vstack([c for c in coeffs_ds])
        except Exception as e:
            # Fall back to alternative implementation if primary fails
            try:
                coeffs_ds = modwt_alternative(ds_signal, wavelet, level=levels)
                wav_signal_ds = np.vstack([c for c in coeffs_ds])
            except:
                # Last resort: use SWT
                coeffs_ds = pywt.swt(ds_signal, wavelet, level=min(levels, pywt.swt_max_level(len(ds_signal))))
                wav_signal_ds = np.vstack([c[1] for c in coeffs_ds])  # Use detail coefficients
        
        # Compute MODWT for SS signal
        try:
            # Use our custom MODWT implementation
            coeffs_ss = modwt(ss_signal, wavelet, level=levels)
            wav_signal_ss = np.vstack([c for c in coeffs_ss])
        except Exception as e:
            # Fall back to alternative implementation if primary fails
            try:
                coeffs_ss = modwt_alternative(ss_signal, wavelet, level=levels)
                wav_signal_ss = np.vstack([c for c in coeffs_ss])
            except:
                # Last resort: use SWT
                coeffs_ss = pywt.swt(ss_signal, wavelet, level=min(levels, pywt.swt_max_level(len(ss_signal))))
                wav_signal_ss = np.vstack([c[1] for c in coeffs_ss])  # Use detail coefficients
        
        # Normalize signals
        wav_signal_ds = self._normalize_range(wav_signal_ds)
        wav_signal_ss = self._normalize_range(wav_signal_ss)
        
        return wav_signal_ds, wav_signal_ss
    
    def _normalize_range(self, data: np.ndarray) -> np.ndarray:
        """
        Normalize data to range [0, 1].
        
        Parameters:
        -----------
        data : np.ndarray
            Input data
            
        Returns:
        --------
        np.ndarray
            Normalized data
        """
        # Handle empty arrays
        if data.size == 0:
            return data
            
        # Calculate min/max for each row
        data_min = np.min(data, axis=1, keepdims=True)
        data_max = np.max(data, axis=1, keepdims=True)
        
        # Handle constant rows
        constant_mask = data_min == data_max
        if np.any(constant_mask):
            normalized = np.zeros_like(data)
            non_constant = ~constant_mask
            if np.any(non_constant):
                normalized[non_constant] = (data[non_constant] - data_min[non_constant]) / (data_max[non_constant] - data_min[non_constant])
            return normalized
        
        # Normalize
        return (data - data_min) / (data_max - data_min)
    
    def _calculate_metrics(self, classification: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Calculate performance metrics from classification results.
        
        Parameters:
        -----------
        classification : np.ndarray
            Array with true labels and predictions
            
        Returns:
        --------
        Tuple[np.ndarray, float]
            - confusion matrix [TP, TN, FP, FN]
            - accuracy
        """
        y_true = classification[:, 0]
        y_pred = classification[:, 1]
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Extract values
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
        else:
            # Handle case where some classes are missing
            tp = fp = tn = fn = 0
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    if i == 1 and j == 1:
                        tp = cm[i, j]
                    elif i == 0 and j == 0:
                        tn = cm[i, j]
                    elif i == 0 and j == 1:
                        fp = cm[i, j]
                    elif i == 1 and j == 0:
                        fn = cm[i, j]
        
        # Calculate accuracy
        accuracy = accuracy_score(y_true, y_pred)
        
        return np.array([tp, tn, fp, fn]), accuracy
    
    def _initialize_results_file(self, filename: str):
        """
        Initialize results CSV file with header.
        
        Parameters:
        -----------
        filename : str
            Output filename (without extension)
        """
        header = [
            'Dataset', 'Wavelet', 'Wavelet level', 'DC Pop Size', 
            'Segment Size', 'Selected Features each Category', 'DCA version',
            'Observations', 'Runtime', 'TP', 'TN', 'FP', 'FN',
            'Precision', 'Sensitivity', 'Specificity', 'Accuracy'
        ]
        
        with open(f"{filename}.csv", 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
    
    def _write_results(self, filename: str, name: str, wavelet: str, p: int,
                      m: int, T: int, n_obs: int, runtime: float,
                      confmat: Optional[np.ndarray], accuracy: Optional[float],
                      error: bool = False):
        """
        Write results to CSV file.
        
        Parameters:
        -----------
        filename : str
            Output filename (without extension)
        name : str
            Dataset name
        wavelet : str
            Wavelet used
        p : int
            Dendritic cell population size
        m : int
            Segment size
        T : int
            Number of features per category
        n_obs : int
            Number of observations
        runtime : float
            Algorithm runtime
        confmat : np.ndarray, optional
            Confusion matrix [TP, TN, FP, FN]
        accuracy : float, optional
            Classification accuracy
        error : bool, optional
            Whether an error occurred (default: False)
        """
        if error or confmat is None:
            # Write error results
            row = [
                name, wavelet, p, p, m, T, "MRA S-dDCA", 
                n_obs, runtime, "ERROR", "ERROR", "ERROR", "ERROR",
                "ERROR", "ERROR", "ERROR", "ERROR"
            ]
        else:
            # Extract confusion matrix values
            tp, tn, fp, fn = confmat
            
            # Calculate additional metrics
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            
            # Create row
            row = [
                name, wavelet, p, p, m, T, "MRA S-dDCA",
                n_obs, runtime, tp, tn, fp, fn,
                precision, sensitivity, specificity, accuracy
            ]
        
        # Write row to file
        with open(f"{filename}.csv", 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row) 