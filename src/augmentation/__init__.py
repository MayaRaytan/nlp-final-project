"""
Augmentation module for drum pattern classification.
Provides data augmentation techniques for MIDI token sequences.
"""

from .augmentation import AugConfig, augment_once, make_balanced_augmented_train

__all__ = ['AugConfig', 'augment_once', 'make_balanced_augmented_train']
