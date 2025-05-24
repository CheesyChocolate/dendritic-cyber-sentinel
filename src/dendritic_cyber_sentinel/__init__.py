# SPDX-FileCopyrightText: 2025-present CheesyChocolate <dev@behnamlal.xyz>
#
# SPDX-License-Identifier: MIT

"""
Dendritic Cyber Sentinel - A network anomaly detection system based on the MRA S-dDCA algorithm.

This package implements the Multi-Resolution Analysis Deterministic Dendritic Cell Algorithm
with Segmentation (MRA S-dDCA) for network anomaly detection, based on the work by 
David Limon-Cantu and Vicente Alarcon-Aquino.

References:
[1] Limon-Cantu, D., & Alarcon-Aquino, V. (2021). Multiresolution dendritic cell algorithm 
    for network anomaly detection. PeerJ Computer Science, 7, e749.
[2] Greensmith, J., & Aickelin, U. (2008). The Deterministic Dendritic Cell Algorithm.
    In Artificial Immune Systems (pp. 291-302).
[3] Gu, F., Greensmith, J., & Aickelin, U. (2009). Integrating Real-Time Analysis with 
    the Dendritic Cell Algorithm through Segmentation. In Proceedings of the 11th Annual 
    Conference on Genetic and Evolutionary Computation (pp. 1203-1210).
"""

from .__about__ import __version__

# Import main classes and functions
from .modules.mra_sdca import MRA_SdDCA
from .modules.feature_selection import FeatureSelector
from .modules.mutual_information import mutual_information
from .modules.data_loaders import (
    KDDLoader, UNSWNB15Loader, 
    prepare_kdd_dataset, prepare_unsw_dataset
)

# Define public API
__all__ = [
    "MRA_SdDCA",
    "FeatureSelector",
    "mutual_information",
    "KDDLoader",
    "UNSWNB15Loader",
    "prepare_kdd_dataset",
    "prepare_unsw_dataset",
]
