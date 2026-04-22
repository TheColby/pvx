"""pvx.integrations — Framework adapters for pvx augmentation pipelines.

Provides thin, zero-dependency-on-framework-in-core adapters so that
pvx augmentation pipelines slot cleanly into PyTorch, HuggingFace
datasets, and TensorFlow data pipelines.

Usage
-----
Each submodule guards its framework import so only the framework you
actually use needs to be installed::

    # PyTorch
    from pvx.integrations.pytorch import PvxAugmentDataset

    # HuggingFace
    from pvx.integrations.huggingface import make_augment_map_fn

    # TensorFlow
    from pvx.integrations.tensorflow import make_tf_augment_fn
"""
