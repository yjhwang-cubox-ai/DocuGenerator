"""
SynthTIGER
Copyright (c) 2021-present NAVER Corp.
MIT license
"""

from synthdocs.components.corpus.base_corpus import BaseCorpus
from synthdocs.components.corpus.char_augmentable_corpus import CharAugmentableCorpus
from synthdocs.components.corpus.length_augmentable_corpus import (
    LengthAugmentableCorpus,
)

__all__ = ["BaseCorpus", "CharAugmentableCorpus", "LengthAugmentableCorpus"]
