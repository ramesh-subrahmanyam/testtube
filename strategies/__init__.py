"""
Trading Strategies

Base classes and concrete strategy implementations.
"""

from .base import (
    BaseStrategy,
    BuyAndHoldStrategy,
    MovingAverageCrossoverStrategy
)
from .vol_normalized_buy_and_hold import VolNormalizedBuyAndHold
from .dma import DMA
from .low_rsi import LowRSI

# Strategy registry for dynamic instantiation
_STRATEGY_REGISTRY = {
    'BuyAndHold': BuyAndHoldStrategy,
    'MovingAverageCrossover': MovingAverageCrossoverStrategy,
    'VolNormalizedBuyAndHold': VolNormalizedBuyAndHold,
    'DMA': DMA,
    'LowRSI': LowRSI,
}


def get_strategy_class(name):
    """
    Get strategy class by name.

    Args:
        name (str): Strategy name

    Returns:
        class: Strategy class

    Raises:
        ValueError: If strategy name not found
    """
    if name not in _STRATEGY_REGISTRY:
        available = ', '.join(_STRATEGY_REGISTRY.keys())
        raise ValueError(f"Strategy '{name}' not found. Available strategies: {available}")
    return _STRATEGY_REGISTRY[name]


__all__ = [
    'BaseStrategy',
    'BuyAndHoldStrategy',
    'MovingAverageCrossoverStrategy',
    'VolNormalizedBuyAndHold',
    'DMA',
    'LowRSI',
    'get_strategy_class',
]
