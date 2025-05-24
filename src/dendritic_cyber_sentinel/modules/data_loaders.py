"""
Data loading module for network intrusion detection datasets.

This module implements data loaders for NSL-KDD and UNSW-NB15 datasets,
based on the MATLAB KDDLoad.m and UNSWNB15Load.m functions.
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Union, Tuple
from pathlib import Path


class KDDLoader:
    """
    Loader for NSL-KDD dataset.
    
    This class implements the functionality of the MATLAB KDDLoad.m function.
    """
    
    def __init__(self):
        """Initialize the KDD loader."""
        self.column_names = [
            "duration", "protocol_type", "service", "flag", "src_bytes", 
            "dst_bytes", "land", "wrong_fragment", "urgent", "hot", 
            "num_failed_logins", "logged_in", "num_compromised", "root_shell", 
            "su_attempted", "num_root", "num_file_creations", "num_shells", 
            "num_access_files", "num_outbound_cmds", "is_host_login", 
            "is_guest_login", "count", "srv_count", "serror_rate", 
            "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate", 
            "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", 
            "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate", 
            "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", 
            "dst_host_serror_rate", "dst_host_srv_serror_rate", 
            "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "cat", "class_id"
        ]
        
        self.numeric_columns = [
            "duration", "src_bytes", "dst_bytes", "land", "wrong_fragment", 
            "urgent", "hot", "num_failed_logins", "logged_in", "num_compromised", 
            "root_shell", "su_attempted", "num_root", "num_file_creations", 
            "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login", 
            "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate", 
            "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate", 
            "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count", 
            "dst_host_same_srv_rate", "dst_host_diff_srv_rate", 
            "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", 
            "dst_host_serror_rate", "dst_host_srv_serror_rate", 
            "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "class_id"
        ]
        
        self.categorical_columns = ["protocol_type", "service", "flag", "cat"]
    
    def load_data(self, filename: Union[str, Path], 
                  data_lines: Optional[List[int]] = None) -> pd.DataFrame:
        """
        Load NSL-KDD dataset from CSV file.
        
        Parameters:
        -----------
        filename : str or Path
            Path to the CSV file
        data_lines : List[int], optional
            Range of lines to read [start, end]. If None, reads all data.
            
        Returns:
        --------
        pd.DataFrame
            Loaded dataset with proper column names and types
        """
        # Set default data lines
        if data_lines is None:
            data_lines = [2, None]  # Skip header, read all data
            
        # Read CSV file
        try:
            if data_lines[1] is None:
                # Read all data starting from data_lines[0]
                df = pd.read_csv(filename, 
                               names=self.column_names,
                               skiprows=data_lines[0]-1 if data_lines[0] > 1 else 0,
                               delimiter=',')
            else:
                # Read specific range
                nrows = data_lines[1] - data_lines[0] + 1
                df = pd.read_csv(filename,
                               names=self.column_names, 
                               skiprows=data_lines[0]-1 if data_lines[0] > 1 else 0,
                               nrows=nrows,
                               delimiter=',')
        except Exception as e:
            raise ValueError(f"Error loading KDD dataset: {e}")
        
        # Convert data types
        for col in self.numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
        for col in self.categorical_columns:
            if col in df.columns:
                df[col] = df[col].astype('category')
        
        return df


class UNSWNB15Loader:
    """
    Loader for UNSW-NB15 dataset.
    
    This class implements the functionality of the MATLAB UNSWNB15Load.m function.
    """
    
    def __init__(self):
        """Initialize the UNSW-NB15 loader."""
        self.column_names = [
            'id', 'dur', 'proto', 'service', 'state', 'spkts', 'dpkts', 
            'sbytes', 'dbytes', 'rate', 'sttl', 'dttl', 'sload', 'dload', 
            'sloss', 'dloss', 'sinpkt', 'dinpkt', 'sjit', 'djit', 'swin', 
            'stcpb', 'dtcpb', 'dwin', 'tcprtt', 'synack', 'ackdat', 'smean', 
            'dmean', 'trans_depth', 'response_body_len', 'ct_srv_src', 
            'ct_state_ttl', 'ct_dst_ltm', 'ct_src_dport_ltm', 'ct_dst_sport_ltm', 
            'ct_dst_src_ltm', 'is_ftp_login', 'ct_ftp_cmd', 'ct_flw_http_mthd', 
            'ct_src_ltm', 'ct_srv_dst', 'is_sm_ips_ports', 'attack_cat', 'label'
        ]
        
        self.numeric_columns = [
            'id', 'dur', 'spkts', 'dpkts', 'sbytes', 'dbytes', 'rate', 'sttl', 
            'dttl', 'sload', 'dload', 'sloss', 'dloss', 'sinpkt', 'dinpkt', 
            'sjit', 'djit', 'swin', 'stcpb', 'dtcpb', 'dwin', 'tcprtt', 
            'synack', 'ackdat', 'smean', 'dmean', 'trans_depth', 
            'response_body_len', 'ct_srv_src', 'ct_state_ttl', 'ct_dst_ltm', 
            'ct_src_dport_ltm', 'ct_dst_sport_ltm', 'ct_dst_src_ltm', 
            'is_ftp_login', 'ct_ftp_cmd', 'ct_flw_http_mthd', 'ct_src_ltm', 
            'ct_srv_dst', 'is_sm_ips_ports', 'label'
        ]
        
        self.categorical_columns = ['proto', 'service', 'state', 'attack_cat']
    
    def load_data(self, filename: Union[str, Path], 
                  start_row: int = 2, end_row: Optional[int] = None) -> pd.DataFrame:
        """
        Load UNSW-NB15 dataset from CSV file.
        
        Parameters:
        -----------
        filename : str or Path
            Path to the CSV file
        start_row : int, optional
            Starting row to read (default: 2, to skip header)
        end_row : int, optional
            Ending row to read. If None, reads to end of file.
            
        Returns:
        --------
        pd.DataFrame
            Loaded dataset with proper column names and types
        """
        try:
            if end_row is None:
                # Read all data starting from start_row
                df = pd.read_csv(filename,
                               names=self.column_names,
                               skiprows=start_row-1 if start_row > 1 else 0,
                               delimiter=',',
                               encoding='utf-8')
            else:
                # Read specific range
                nrows = end_row - start_row + 1
                df = pd.read_csv(filename,
                               names=self.column_names,
                               skiprows=start_row-1 if start_row > 1 else 0,
                               nrows=nrows,
                               delimiter=',',
                               encoding='utf-8')
        except Exception as e:
            raise ValueError(f"Error loading UNSW-NB15 dataset: {e}")
        
        # Convert data types
        for col in self.numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
        for col in self.categorical_columns:
            if col in df.columns:
                df[col] = df[col].astype('category')
        
        # Handle missing values
        df = df.fillna(0)  # Fill NaN with 0 for numeric columns
        
        return df


def prepare_kdd_dataset(dataset: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare NSL-KDD dataset for MRA S-dDCA processing.
    
    This function replicates the data preparation steps from 
    MRA_SdDCA_test_NSL_KDD.m
    
    Parameters:
    -----------
    dataset : pd.DataFrame
        Raw NSL-KDD dataset
        
    Returns:
    --------
    pd.DataFrame
        Prepared dataset ready for MRA S-dDCA algorithm
    """
    # Create a copy to avoid modifying the original
    prepared = dataset.copy()
    
    # Create newid_addon column
    prepared['newid_addon'] = (
        prepared['protocol_type'].astype(str) + '_' + 
        prepared['service'].astype(str) + '_' + 
        prepared['flag'].astype(str)
    )
    
    # Create binary dataset_label (0=normal, 1=anomaly)
    prepared['dataset_label'] = (prepared['cat'] != 'normal').astype(int)
    
    # Keep the original attack type for multi-class classification
    prepared['attack_type'] = prepared['cat']
    prepared['attack_class'] = prepared['class_id']
    
    # Remove categorical columns that will be encoded in newid_addon
    prepared = prepared.drop(['protocol_type', 'service', 'flag'], axis=1)
    
    return prepared


def prepare_unsw_dataset(dataset: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare UNSW-NB15 dataset for MRA S-dDCA processing.
    
    This function replicates the data preparation steps from
    MRA_SdDCA_test_UNSW_NB15.m
    
    Parameters:
    -----------
    dataset : pd.DataFrame
        Raw UNSW-NB15 dataset
        
    Returns:
    --------
    pd.DataFrame
        Prepared dataset ready for MRA S-dDCA algorithm
    """
    # Create a copy to avoid modifying the original
    prepared = dataset.copy()
    
    # Create newid_addon column
    prepared['newid_addon'] = (
        prepared['proto'].astype(str) + '_' + 
        prepared['service'].astype(str) + '_' + 
        prepared['state'].astype(str)
    )
    
    # Rename 'label' to 'dataset_label' for consistency
    prepared = prepared.rename(columns={'label': 'dataset_label'})
    
    # Create 'cat' column from 'attack_cat' for consistency with KDD format
    # If attack_cat is empty but dataset_label is 1, use 'generic'
    prepared['cat'] = prepared['attack_cat']
    mask = (prepared['dataset_label'] == 1) & (prepared['attack_cat'].isnull() | (prepared['attack_cat'] == ''))
    prepared.loc[mask, 'cat'] = 'generic'
    prepared.loc[prepared['dataset_label'] == 0, 'cat'] = 'normal'
    
    # Remove categorical columns that will be encoded in newid_addon
    prepared = prepared.drop(['proto', 'service', 'state'], axis=1)
    
    return prepared


def load_nsl_kdd(filepath: Union[str, Path], verbose: bool = False) -> pd.DataFrame:
    """
    Load and prepare NSL-KDD dataset for MRA S-dDCA algorithm.
    
    Parameters:
    -----------
    filepath : str or Path
        Path to the NSL-KDD dataset CSV file
    verbose : bool, optional
        Whether to print progress information
        
    Returns:
    --------
    pd.DataFrame
        Prepared dataset ready for MRA S-dDCA algorithm
    """
    if verbose:
        print(f"Loading dataset from: {filepath}")
    
    # Load raw dataset
    loader = KDDLoader()
    dataset = loader.load_data(filepath)
    
    if verbose:
        print(f"Loaded dataset with {len(dataset)} samples")
    
    # Prepare dataset
    prepared_dataset = prepare_kdd_dataset(dataset)
    
    if verbose:
        print(f"Prepared dataset with {len(prepared_dataset)} samples and {len(prepared_dataset.columns)} features")
    
    return prepared_dataset


def load_unsw_nb15(filepath: Union[str, Path], verbose: bool = False) -> pd.DataFrame:
    """
    Load and prepare UNSW-NB15 dataset for MRA S-dDCA algorithm.
    
    Parameters:
    -----------
    filepath : str or Path
        Path to the UNSW-NB15 dataset CSV file
    verbose : bool, optional
        Whether to print progress information
        
    Returns:
    --------
    pd.DataFrame
        Prepared dataset ready for MRA S-dDCA algorithm
    """
    if verbose:
        print(f"Loading dataset from: {filepath}")
    
    # Load raw dataset
    loader = UNSWNB15Loader()
    dataset = loader.load_data(filepath)
    
    if verbose:
        print(f"Loaded dataset with {len(dataset)} samples")
    
    # Prepare dataset
    prepared_dataset = prepare_unsw_dataset(dataset)
    
    if verbose:
        print(f"Prepared dataset with {len(prepared_dataset)} samples and {len(prepared_dataset.columns)} features")
    
    return prepared_dataset 