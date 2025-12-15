"""
Constant Exposure Buy and Hold Strategy

This strategy always maintains a long position (position = 1), and when used
with the Backtester with ConstantDollarExposure, maintains a constant dollar
exposure regardless of price or volatility changes.

This provides a comparison benchmark for evaluating tactical strategies against
a traditional buy-and-hold approach with constant dollar exposure.
"""

import pandas as pd
import logging
from .base import BaseStrategy


logger = logging.getLogger(__name__)


class ConstantExposureBuyAndHold(BaseStrategy):
    """
    Constant Exposure Buy and Hold Strategy.

    This strategy always holds a long position (signal = 1 every day).

    Key Characteristics:
    - Position is always 1 (never 0 or -1)
    - When used with Backtester with ConstantDollarExposure, maintains constant dollar exposure
    - Provides comparison to constant-dollar buy-and-hold investing

    Difference from volatility-normalized Buy & Hold:
    - Vol-normalized B&H: Position adjusted to maintain constant dollar volatility
    - Constant exposure B&H: Position adjusted to maintain constant dollar exposure

    Example:
        from strategies.constant_exposure_buy_and_hold import ConstantExposureBuyAndHold
        from libs.backtester import Backtester
        from libs.exposure_management import ConstantDollarExposure

        # Create strategy
        strategy = ConstantExposureBuyAndHold()

        # Create exposure manager
        exp_mgr = ConstantDollarExposure(target_dollar_exposure=100000)

        # Run backtest
        backtester = Backtester(strategy, exposure_manager=exp_mgr)
        backtester('AAPL', '2020-01-01', '2024-12-31', slippage_bps=5)

        # Compare with DMA
        print(f"B&H Sharpe: {backtester.slipped_performance['sharpe']:.3f}")
    """

    name = "ConstantExposureBuyAndHold"

    def __init__(self):
        """
        Initialize the Constant Exposure Buy and Hold strategy.

        Note:
            This strategy has no configurable parameters,
            since it always holds position = 1.
        """
        # Initialize base class with no strategy-specific parameters
        super().__init__()

        logger.info("Initialized ConstantExposureBuyAndHold strategy")
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
            - The Backtester will apply constant dollar exposure sizing
        """
        df = prices.copy()

        # Always long (signal = 1) for every date
        df['Signal'] = 1
        df['Position'] = 1

        # Return only the strategy period
        result = df.loc[self.start_date:self.end_date]

        logger.info(f"Generated {len(result)} signals for Constant Exposure Buy & Hold strategy")
        logger.info(f"All positions = 1 (always long)")

        return result


if __name__ == "__main__":
    # Example usage and comparison
    import sys
    from pathlib import Path

    # Add parent directory to path
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from libs.backtester import Backtester
    from libs.exposure_management import ConstantDollarExposure
    import logging

    logging.basicConfig(level=logging.INFO)

    print("=" * 80)
    print("Constant Exposure Buy & Hold Example")
    print("=" * 80)
    print()

    symbol = 'AAPL'
    start_date = '2024-01-01'
    end_date = '2024-12-31'
    target_exposure = 100000

    # Test Constant Exposure Buy & Hold strategy
    print("Running Constant Exposure Buy & Hold backtest...")
    print("-" * 80)

    strategy = ConstantExposureBuyAndHold()
    exp_mgr = ConstantDollarExposure(target_dollar_exposure=target_exposure)
    backtester = Backtester(strategy, exposure_manager=exp_mgr)
    backtester(symbol, start_date, end_date, slippage_bps=5)

    print()
    print("CONSTANT EXPOSURE BUY & HOLD RESULTS:")
    print(f"  Total PnL:    ${backtester.slipped_performance['total_pnl']:,.2f}")
    print(f"  Sharpe Ratio: {backtester.slipped_performance['sharpe']:.3f}")
    print(f"  Num Trades:   {backtester.slipped_performance['num_trades']:,}")

    print()
    print("=" * 80)
