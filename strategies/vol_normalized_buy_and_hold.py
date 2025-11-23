"""
Volatility-Normalized Buy and Hold Strategy

This strategy always maintains a long position (position = 1), but when used
with the Backtester, the position is volatility-adjusted to match the risk
profile of other volatility-normalized strategies.

This provides an apples-to-apples comparison benchmark for evaluating tactical
strategies like DMA against a passive buy-and-hold approach with the same
volatility normalization methodology.
"""

import pandas as pd
import logging
from .base import BaseStrategy


logger = logging.getLogger(__name__)


class VolNormalizedBuyAndHold(BaseStrategy):
    """
    Volatility-Normalized Buy and Hold Strategy.

    This strategy always holds a long position (signal = 1 every day).

    Key Characteristics:
    - Position is always 1 (never 0 or -1)
    - When used with Backtester, position is scaled by (dollar_size / dollar_volatility)
    - Provides fair comparison to other vol-normalized strategies
    - Same risk profile (dollar volatility) as tactical strategies

    Difference from traditional Buy & Hold:
    - Traditional B&H: Buy once, position size grows/shrinks with price
    - Vol-normalized B&H: Position continuously adjusted to maintain constant dollar volatility

    Example:
        from strategies.vol_normalized_buy_and_hold import VolNormalizedBuyAndHold
        from libs.backtester import Backtester

        # Create strategy
        strategy = VolNormalizedBuyAndHold()

        # Run backtest
        backtester = Backtester(strategy, dollar_size=100000)
        backtester('AAPL', '2020-01-01', '2024-12-31', slippage_bps=5)

        # Compare with DMA
        print(f"B&H Sharpe: {backtester.slipped_performance['sharpe']:.3f}")
    """

    def __init__(self, volatility_function=None):
        """
        Initialize the Volatility-Normalized Buy and Hold strategy.

        Args:
            volatility_function (callable, optional): Function to calculate volatility.
                                                     Default: simple_vol with N=20

        Note:
            This strategy has no configurable parameters beyond the volatility function,
            since it always holds position = 1.
        """
        # Initialize base class with no strategy-specific parameters
        super().__init__(volatility_function=volatility_function)

        logger.info("Initialized VolNormalizedBuyAndHold strategy")
        logger.info("Position: Always long (1)")

    def generate_signals(self, prices):
        """
        Generate trading signals - always returns 1 (long).

        Args:
            prices (pd.DataFrame): Price data with OHLCV columns

        Returns:
            pd.DataFrame: DataFrame with Close, Signal, and Position columns
                         All signals and positions are 1 (long)

        Implementation:
            - Creates signals = 1 for every date in the price data
            - This ensures the strategy is always fully invested
            - The Backtester will apply volatility normalization
        """
        df = prices.copy()

        # Always long (signal = 1) for every date
        df['Signal'] = 1
        df['Position'] = 1

        # Return only the strategy period
        result = df.loc[self.start_date:self.end_date]

        logger.info(f"Generated {len(result)} signals for Buy & Hold strategy")
        logger.info(f"All positions = 1 (always long)")

        return result


# For backward compatibility, create an alias
BuyAndHold = VolNormalizedBuyAndHold


if __name__ == "__main__":
    # Example usage and comparison
    import sys
    from pathlib import Path

    # Add parent directory to path
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from libs.backtester import Backtester
    from strategies.dma import DMA
    from functools import partial
    from libs.volatility import simple_vol
    import logging

    logging.basicConfig(level=logging.INFO)

    print("=" * 80)
    print("Volatility-Normalized Buy & Hold vs DMA Comparison")
    print("=" * 80)
    print()

    symbol = 'AAPL'
    start_date = '2020-01-01'
    end_date = '2024-12-31'
    dollar_size = 100000
    slippage_bps = 5

    # Test Buy & Hold strategy
    print("Running Volatility-Normalized Buy & Hold backtest...")
    print("-" * 80)

    bnh_strategy = VolNormalizedBuyAndHold(
        volatility_function=partial(simple_vol, N=20)
    )
    bnh_backtester = Backtester(bnh_strategy, dollar_size=dollar_size)
    bnh_backtester(symbol, start_date, end_date, slippage_bps=slippage_bps)

    print()
    print("BUY & HOLD RESULTS:")
    print(f"  Total PnL:    ${bnh_backtester.slipped_performance['total_pnl']:,.2f}")
    print(f"  Sharpe Ratio: {bnh_backtester.slipped_performance['sharpe']:.3f}")
    print(f"  Num Trades:   {bnh_backtester.slipped_performance['num_trades']:,}")

    # Test DMA strategy
    print()
    print("Running 200-DMA backtest...")
    print("-" * 80)

    dma_strategy = DMA(lookback=200)
    dma_backtester = Backtester(dma_strategy, dollar_size=dollar_size)
    dma_backtester(symbol, start_date, end_date, slippage_bps=slippage_bps)

    print()
    print("DMA RESULTS:")
    print(f"  Total PnL:    ${dma_backtester.slipped_performance['total_pnl']:,.2f}")
    print(f"  Sharpe Ratio: {dma_backtester.slipped_performance['sharpe']:.3f}")
    print(f"  Num Trades:   {dma_backtester.slipped_performance['num_trades']:,}")

    # Comparison
    print()
    print("=" * 80)
    print("COMPARISON (DMA vs Buy & Hold)")
    print("=" * 80)

    bnh_pnl = bnh_backtester.slipped_performance['total_pnl']
    dma_pnl = dma_backtester.slipped_performance['total_pnl']
    pnl_diff = dma_pnl - bnh_pnl
    pnl_diff_pct = (pnl_diff / abs(bnh_pnl) * 100) if bnh_pnl != 0 else 0

    bnh_sharpe = bnh_backtester.slipped_performance['sharpe']
    dma_sharpe = dma_backtester.slipped_performance['sharpe']
    sharpe_diff = dma_sharpe - bnh_sharpe

    print(f"PnL Difference:       ${pnl_diff:,.2f} ({pnl_diff_pct:+.2f}%)")
    print(f"Sharpe Difference:    {sharpe_diff:+.3f}")
    print()

    if dma_pnl > bnh_pnl:
        print(f"✓ DMA outperformed Buy & Hold by ${pnl_diff:,.2f}")
    else:
        print(f"✗ DMA underperformed Buy & Hold by ${-pnl_diff:,.2f}")

    if dma_sharpe > bnh_sharpe:
        print(f"✓ DMA has better risk-adjusted returns (Sharpe: {sharpe_diff:+.3f})")
    else:
        print(f"✗ Buy & Hold has better risk-adjusted returns (Sharpe: {sharpe_diff:+.3f})")

    print()
    print("=" * 80)
    print("Key Insight:")
    print("  Both strategies use identical volatility normalization methodology.")
    print("  The difference shows the value-add of DMA timing vs. being always long.")
    print("=" * 80)
