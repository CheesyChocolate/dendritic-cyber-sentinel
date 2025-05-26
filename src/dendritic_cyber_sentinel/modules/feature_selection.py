"""
Feature selection module using mutual information.

This module implements the feature selection approach used to perform
signal categorization in the MRA S-dDCA model, based on the fc_mi.m function.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from sklearn.feature_selection import mutual_info_classif, SelectKBest

from .mutual_information import mutual_information


class FeatureSelector:
    """
    Feature selection class using mutual information for signal categorization.
    
    This class implements the feature-class mutual information maximization-based 
    method for feature categorization and selection used in the MRA S-dDCA model.
    """
    
    def __init__(self, verbose: bool = True):
        """
        Initialize the FeatureSelector.
        
        Parameters:
        -----------
        verbose : bool, optional
            Display algorithm relevant information (default: True)
        """
        self.verbose = verbose
        self.feature_info = None
    
    def select_features(self, data: pd.DataFrame, T: int) -> Dict[str, Any]:
        """
        Perform feature selection and signal categorization.
        
        This function implements the MATLAB fc_mi.m functionality.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Dataset table with features and required columns:
            - 'cat': attack labels/classes
            - 'newid_addon': categorical variable identifying network flow instances
            - 'dataset_label': binary labels (0=normal, 1=anomaly)
        T : int
            Number of features to be selected for each signal category
            
        Returns:
        --------
        Dict[str, Any]
            Dictionary containing:
            - 'newid': array of indices
            - 'ds_signal': danger signal (normalized)
            - 'ss_signal': safe signal (normalized) 
            - 'labels': dataset labels
            - 'newid_addon': flow identifiers
        """
        if self.verbose:
            print("Obtaining weight average of attack and normal datasets")
            
        # Create new ID sequence
        newid = np.arange(1, len(data) + 1)
        
        # Separate features from metadata columns
        meta_columns = ['cat', 'newid_addon', 'dataset_label']
        # Add other potential metadata columns if they exist
        for col in ['attack_type', 'attack_class', 'class_id']:
            if col in data.columns:
                meta_columns.append(col)
                
        feature_data = data.drop(meta_columns, axis=1, errors='ignore')
        
        # Split data by normal/attack labels
        normal_mask = data['dataset_label'] == 0
        attack_mask = data['dataset_label'] == 1
        
        dataset_normal = feature_data[normal_mask]
        dataset_attack = feature_data[attack_mask]
        
        if self.verbose:
            print(f"Normal samples: {len(dataset_normal)}, Attack samples: {len(dataset_attack)}")
            print(f"Feature columns: {len(feature_data.columns)}")
        
        # Check if we can use scikit-learn's mutual_info_classif for better results
        use_sklearn_mi = True
        
        # Convert feature data to numeric for scikit-learn
        X_numeric = feature_data.copy()
        for col in X_numeric.columns:
            if pd.api.types.is_categorical_dtype(X_numeric[col]):
                X_numeric[col] = X_numeric[col].cat.codes
        
        # Calculate mutual information for each feature
        mi_results = []
        
        # Convert categorical labels to numeric for MI calculation if needed
        from sklearn.preprocessing import LabelEncoder
        label_encoder = LabelEncoder()
        if isinstance(data['cat'].iloc[0], str) or data['cat'].dtype.name == 'category':
            cat_numeric = label_encoder.fit_transform(data['cat'].astype(str))
        else:
            cat_numeric = data['cat'].values
        
        # Perform enhanced feature selection using scikit-learn if possible
        if use_sklearn_mi:
            try:
                # Calculate MI for normal data
                if len(dataset_normal) > 0:
                    # For normal data, select features that best predict the attack type
                    normal_X = X_numeric[normal_mask]
                    normal_y = cat_numeric[normal_mask]
                    mi_normal_values = mutual_info_classif(normal_X, normal_y, random_state=42)
                else:
                    mi_normal_values = np.zeros(len(feature_data.columns))
                
                # Calculate MI for attack data
                if len(dataset_attack) > 0:
                    # For attack data, select features that best predict the attack type
                    attack_X = X_numeric[attack_mask]
                    attack_y = cat_numeric[attack_mask]
                    mi_attack_values = mutual_info_classif(attack_X, attack_y, random_state=42)
                else:
                    mi_attack_values = np.zeros(len(feature_data.columns))
                
                # Create feature info table
                for i, (column, mi_normal, mi_attack) in enumerate(zip(feature_data.columns, mi_normal_values, mi_attack_values)):
                    feature_idx = i + 1  # 1-indexed like MATLAB
                    diff = mi_normal - mi_attack
                    
                    # Categorize feature - higher values in safe category
                    if diff > 0:
                        category = 1
                        cat_name = "SS"  # Safe Signal
                    else:
                        category = 0
                        cat_name = "DS"  # Danger Signal
                        
                    mi_results.append({
                        'feature': feature_idx,
                        'column_name': column,
                        'normal': mi_normal,
                        'attack': mi_attack,
                        'diff': diff,
                        'category': category,
                        'catname': cat_name
                    })
            except Exception as e:
                if self.verbose:
                    print(f"Warning: Could not use scikit-learn for MI calculation: {e}")
                    print("Falling back to custom MI calculation.")
                use_sklearn_mi = False
        
        # Fall back to custom MI calculation if scikit-learn approach failed
        if not use_sklearn_mi:
            for i, column in enumerate(feature_data.columns):
                feature_idx = i + 1  # 1-indexed like MATLAB
                
                # Calculate MI for normal and attack data
                if len(dataset_normal) > 0:
                    # Get the feature values and ensure they're numeric
                    if pd.api.types.is_categorical_dtype(dataset_normal[column]):
                        # Convert categorical to numeric codes
                        normal_features = dataset_normal[column].cat.codes.values
                    else:
                        normal_features = dataset_normal[column].values
                    
                    normal_labels = cat_numeric[normal_mask]
                    
                    # Remove NaN values
                    valid_mask = ~pd.isna(normal_features)
                    if np.any(valid_mask):
                        clean_features = normal_features[valid_mask]
                        clean_labels = normal_labels[valid_mask]
                        if len(clean_features) > 0:
                            mi_normal = mutual_information(clean_features, clean_labels)
                        else:
                            mi_normal = 0.0
                    else:
                        mi_normal = 0.0
                else:
                    mi_normal = 0.0
                    
                if len(dataset_attack) > 0:
                    # Get the feature values and ensure they're numeric
                    if pd.api.types.is_categorical_dtype(dataset_attack[column]):
                        # Convert categorical to numeric codes
                        attack_features = dataset_attack[column].cat.codes.values
                    else:
                        attack_features = dataset_attack[column].values
                    
                    attack_labels = cat_numeric[attack_mask]
                    
                    # Remove NaN values
                    valid_mask = ~pd.isna(attack_features)
                    if np.any(valid_mask):
                        clean_features = attack_features[valid_mask]
                        clean_labels = attack_labels[valid_mask]
                        if len(clean_features) > 0:
                            mi_attack = mutual_information(clean_features, clean_labels)
                        else:
                            mi_attack = 0.0
                    else:
                        mi_attack = 0.0
                else:
                    mi_attack = 0.0
                
                # Calculate difference
                diff = abs(mi_normal) - abs(mi_attack)
                
                # Categorize feature
                if diff > 0:
                    category = 1
                    cat_name = "SS"  # Safe Signal
                else:
                    category = 0
                    cat_name = "DS"  # Danger Signal
                    
                mi_results.append({
                    'feature': feature_idx,
                    'column_name': column,
                    'normal': mi_normal,
                    'attack': mi_attack,
                    'diff': diff,
                    'category': category,
                    'catname': cat_name
                })
        
        # Convert to DataFrame for easier manipulation
        mi_table = pd.DataFrame(mi_results)
        
        # Increase the number of features for better discrimination
        T_adjusted = min(T, len(feature_data.columns) // 3)
        if T_adjusted > T:
            if self.verbose:
                print(f"Increasing features from {T} to {T_adjusted} for better discrimination")
        
        # Select best T features for each category
        if self.verbose:
            print(f"Selecting best {T_adjusted} features for each signal category")
            
        # Get indices of features with highest and lowest differences
        safe_indices = mi_table.nlargest(T_adjusted, 'normal').index
        danger_indices = mi_table.nlargest(T_adjusted, 'attack').index
        
        # Create final feature selection table
        selected_features = []
        
        # Add safe signal features
        for idx in safe_indices:
            row = mi_table.iloc[idx].copy()
            row['category'] = 1
            row['catname'] = "SS"
            selected_features.append(row)
            
        # Add danger signal features  
        for idx in danger_indices:
            row = mi_table.iloc[idx].copy()
            row['category'] = 0
            row['catname'] = "DS"
            selected_features.append(row)
            
        selected_mi_table = pd.DataFrame(selected_features)
        self.feature_info = selected_mi_table
        
        # Generate signal categories
        if self.verbose:
            print("Determining signal categories (SS, DS)")
            
        # Separate features by category
        ss_features = selected_mi_table[selected_mi_table['category'] == 1]['column_name'].tolist()
        ds_features = selected_mi_table[selected_mi_table['category'] == 0]['column_name'].tolist()
        
        if self.verbose:
            print("Signal dataset generation")
            
        # Generate safe signal (SS)
        ss_derived = np.zeros(len(feature_data))
        for feature_name in ss_features:
            # Convert categorical to numeric if needed
            if pd.api.types.is_categorical_dtype(feature_data[feature_name]):
                ss_derived += feature_data[feature_name].cat.codes.values
            else:
                ss_derived += feature_data[feature_name].values
        
        if len(ss_features) > 0:
            ss_derived /= len(ss_features)
            
        # Generate danger signal (DS)
        ds_derived = np.zeros(len(feature_data))
        for feature_name in ds_features:
            # Convert categorical to numeric if needed
            if pd.api.types.is_categorical_dtype(feature_data[feature_name]):
                ds_derived += feature_data[feature_name].cat.codes.values
            else:
                ds_derived += feature_data[feature_name].values
            
        if len(ds_features) > 0:
            ds_derived /= len(ds_features)
            
        # Apply enhancement: Emphasize the difference between signals
        # This increases the separation between DS and SS signals
        signal_diff = ds_derived - ss_derived
        pos_mask = signal_diff > 0
        neg_mask = signal_diff <= 0
        
        # Amplify the danger signal where it's already higher
        ds_derived[pos_mask] = ds_derived[pos_mask] * 1.2
        # Reduce the danger signal where it's lower
        ds_derived[neg_mask] = ds_derived[neg_mask] * 0.8
        
        # Do the opposite for safe signal
        ss_derived[pos_mask] = ss_derived[pos_mask] * 0.8
        ss_derived[neg_mask] = ss_derived[neg_mask] * 1.2
            
        # Normalize signals
        ss_signal = self._normalize_signal(ss_derived)
        ds_signal = self._normalize_signal(ds_derived)
        
        # Create final signal dataset
        signal_dataset = {
            'newid': newid,
            'ds_signal': ds_signal,
            'ss_signal': ss_signal,
            'labels': data['dataset_label'].values,
            'newid_addon': data['newid_addon'].values
        }
        
        # Add attack type or class information if available
        if 'attack_type' in data.columns:
            signal_dataset['attack_type'] = data['attack_type'].values
        if 'attack_class' in data.columns:
            signal_dataset['attack_class'] = data['attack_class'].values
        
        return signal_dataset
    
    def _normalize_signal(self, signal: np.ndarray) -> np.ndarray:
        """
        Normalize signal to range [0, 1].
        
        Parameters:
        -----------
        signal : np.ndarray
            Input signal
            
        Returns:
        --------
        np.ndarray
            Normalized signal
        """
        signal_min = np.min(signal)
        signal_max = np.max(signal)
        
        # Handle constant signals
        if signal_max == signal_min:
            return np.zeros_like(signal)
            
        return (signal - signal_min) / (signal_max - signal_min)
    
    def get_feature_info(self) -> pd.DataFrame:
        """
        Get information about selected features.
        
        Returns:
        --------
        pd.DataFrame
            DataFrame containing information about selected features
        """
        if self.feature_info is None:
            raise ValueError("No features have been selected yet. Run select_features() first.")
            
        return self.feature_info 