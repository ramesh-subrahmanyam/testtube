"""
Backtester Module

This module provides the Backtester class to run trading strategies
with exposure-adjusted position sizing and slippage modeling.
"""

import pandas as pd
import numpy as np
import logging
from .performance import stats


logger = logging.getLogger(__name__)


class Backtester:
    """
    Backtester class to run trading strategies with exposure-adjusted position sizing.

    The Backtester:
    1. Runs a strategy on a symbol/date range
    2. Uses an ExposureManager to calculate position sizing multipliers
    3. Adjusts positions by multiplying raw positions by the multiplier
    4. Calculates unslipped and slipped PnL
    5. Applies slippage based on position changes and slippage_bps
    6. Computes performance statistics for both

    All calculations are fully vectorized for efficiency.
    """

    def __init__(self, strategy, dollar_size=100000, exposure_manager=None):
        """
        Initialize the Backtester.

        Args:
            strategy: Strategy object (instance of BaseStrategy or subclass)
            dollar_size (float): Target dollar size for position sizing (default: 100,000)
                                 Note: Only used if exposure_manager is None
            exposure_manager (ExposureManager, optional): ExposureManager object for
                                                          exposure-adjusted position sizing.
                                                          If provided, overrides dollar_size.

        Example:
            from strategies import MovingAverageCrossoverStrategy
            from libs.backtester import Backtester
            from libs.exposure_management import VolatilityTargetExposure, ConstantDollarExposure
            from functools import partial
            from libs.volatility import simple_vol

            # Without exposure_manager (legacy)
            strategy = MovingAverageCrossoverStrategy(short_period=20, long_period=50)
            backtester = Backtester(strategy, dollar_size=100000)
            backtester('AAPL', '2024-01-01', '2024-12-31', slippage_bps=5)

            # With VolatilityTargetExposure (recommended)
            exp_mgr = VolatilityTargetExposure(
                total_dollar_size=100000,
                volatility_function=partial(simple_vol, N=20)
            )
            backtester = Backtester(strategy, exposure_manager=exp_mgr)
            backtester('AAPL', '2024-01-01', '2024-12-31', slippage_bps=5)

            # With ConstantDollarExposure
            exp_mgr = ConstantDollarExposure(target_dollar_exposure=100000)
            backtester = Backtester(strategy, exposure_manager=exp_mgr)

            # With periodic rebalancing
            exp_mgr = VolatilityTargetExposure(100000).periodic(
                frequency="monthly",
                when="start"
            )
            backtester = Backtester(strategy, exposure_manager=exp_mgr)
        """
        self.strategy = strategy
        self.dollar_size = dollar_size
        self.exposure_manager = exposure_manager

        # Will be set when __call__ is invoked
        self.symbol = None
        self.start_date = None
        self.end_date = None
        self.slippage_bps = None
        self.slipped_performance = None
        self.unslipped_performance = None

        logger.info(f"Initialized Backtester with {strategy.__class__.__name__}")
        if exposure_manager is not None:
            logger.info(f"Using ExposureManager: {exposure_manager}")
        else:
            logger.info(f"Dollar size: ${dollar_size:,.2f}")

    def __call__(self, symbol, start_date, end_date, slippage_bps=0):
        """
        Run the strategy and calculate exposure-adjusted PnL with slippage.

        Args:
            symbol (str): Stock ticker symbol
            start_date (str|datetime): Start date (YYYY-MM-DD)
            end_date (str|datetime): End date (YYYY-MM-DD)
            slippage_bps (float): Slippage in basis points (default: 0)
                                 Example: 5 bps = 0.05% per trade

        Returns:
            None (results stored in self attributes)

        Attributes set:
            - slipped_performance: Performance stats with slippage
            - unslipped_performance: Performance stats without slippage

        Example:
            backtester('AAPL', '2024-01-01', '2024-12-31', slippage_bps=5)
            print(backtester.slipped_performance)
            print(backtester.unslipped_performance)
        """
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.slippage_bps = slippage_bps

        logger.info(f"Running backtest on {symbol} from {start_date} to {end_date}")
        logger.info(f"Slippage: {slippage_bps} bps")

        # Step 1: Run strategy
        logger.info("Running strategy...")
        self.strategy(symbol, start_date, end_date)

        if self.strategy.df is None or self.strategy.df.empty:
            logger.error(f"Strategy execution failed for {symbol}")
            self.slipped_performance = None
            self.unslipped_performance = None
            return

        df = self.strategy.df.copy()

        # Step 2 & 3: Calculate exposure-adjusted positions
        if self.exposure_manager is not None:
            # Use ExposureManager object to calculate multiplier
            logger.info("Using ExposureManager to calculate exposure multiplier...")
            multiplier = self.exposure_manager(df)
            df['Exposure_Multiplier'] = multiplier

            # Exposure-adjusted position = position × multiplier
            df['Exposure_Adjusted_Position'] = df['Position'] * multiplier

            # Calculate dollar volatility for reporting purposes
            df['Dollar_Volatility'] = (df['Volatility'] / 100) * df['Close']
        else:
            # Legacy behavior: Calculate dollar volatility and vol-adjusted positions
            # Dollar volatility = (return volatility / 100) × price
            # Volatility is already in percentage form from volatility functions
            logger.info("Calculating dollar volatility...")
            df['Dollar_Volatility'] = (df['Volatility'] / 100) * df['Close']

            # Handle division by zero or NaN
            df['Dollar_Volatility'] = df['Dollar_Volatility'].replace(0, np.nan)

            # Calculate vol-adjusted positions (vectorized)
            # Position = position × (dollar_size / dollar_volatility)
            logger.info("Calculating exposure-adjusted positions...")
            df['Exposure_Multiplier'] = self.dollar_size / df['Dollar_Volatility']
            df['Exposure_Adjusted_Position'] = df['Position'] * df['Exposure_Multiplier']

            # Fill NaN positions with 0
            df['Exposure_Adjusted_Position'] = df['Exposure_Adjusted_Position'].fillna(0)

        # Step 4: Calculate unslipped PnL (vectorized)
        # Daily return
        df['Return'] = df['Close'].pct_change()

        # Lag positions by 1 day (today's position applies to tomorrow's return)
        df['Lagged_Position'] = df['Exposure_Adjusted_Position'].shift(1).fillna(0)

        # Check if strategy uses enter_at_open, enter_at_next_close, or exit_at_open
        enter_at_open = hasattr(self.strategy, '_enter_at_open') and self.strategy._enter_at_open
        enter_at_next_close = hasattr(self.strategy, '_enter_at_next_close') and self.strategy._enter_at_next_close
        exit_at_open = hasattr(self.strategy, '_exit_at_open') and self.strategy._exit_at_open

        # Validate: cannot use both enter_at_next_close and enter_at_open
        if enter_at_next_close and enter_at_open:
            logger.error("Cannot use both enter_at_next_close() and enter_at_open() together")
            raise ValueError("Cannot use both enter_at_next_close() and enter_at_open() together. Choose one entry method.")

        if enter_at_open or enter_at_next_close or exit_at_open:
            if enter_at_open:
                logger.info("Using enter_at_open: entries at next day's open price")
            if enter_at_next_close:
                logger.info("Using enter_at_next_close: entries at next day's close price (lag=1)")
            if exit_at_open:
                logger.info("Using exit_at_open: exits at next day's open price")

            # Detect entry and exit points
            lagged_position_was_zero = df['Lagged_Position'].shift(1).fillna(0) == 0
            lagged_position_is_nonzero = df['Lagged_Position'] != 0
            lagged_position_was_nonzero = df['Lagged_Position'].shift(1).fillna(0) != 0
            lagged_position_is_zero = df['Lagged_Position'] == 0

            is_entry = lagged_position_was_zero & lagged_position_is_nonzero
            is_exit = lagged_position_was_nonzero & lagged_position_is_zero

            # Calculate price change: default is Close[T] - Close[T-1]
            df['Price_Change'] = df['Close'].diff()

            # Adjust entry days if enter_at_open
            if enter_at_open:
                # Entry PnL = lagged_position × (Close[T] - Open[T])
                df.loc[is_entry, 'Price_Change'] = df.loc[is_entry, 'Close'] - df.loc[is_entry, 'Open']

            # Adjust entry days if enter_at_next_close
            if enter_at_next_close:
                # Entry happens at next close (T+1) instead of current close (T)
                # So on the entry day T, we have NO PnL (Price_Change = 0)
                # The first PnL happens on day T+1: Close[T+1] - Close[T]
                # But since lagged_position is already shifted, on day T+1 we have the position
                # and the normal calculation gives us: position × (Close[T+1] - Close[T])
                # So we just need to zero out the PnL on the entry signal day (T)
                df.loc[is_entry, 'Price_Change'] = 0

            # Adjust exit days if exit_at_open
            if exit_at_open:
                # Exit PnL = lagged_position × (Open[T] - Close[T-1])
                # Find where the ACTUAL position (not lagged) goes to zero
                actual_position_was_nonzero = df['Exposure_Adjusted_Position'].shift(1).fillna(0) != 0
                actual_position_is_zero = df['Exposure_Adjusted_Position'] == 0
                is_exit_day = actual_position_was_nonzero & actual_position_is_zero

                # On the exit day, we extend PnL to the next open
                # Adjust price change for exit days to use next open: Open[T] - Close[T-1]
                for idx in df.index[is_exit_day]:
                    idx_loc = df.index.get_loc(idx)
                    if idx_loc > 0:
                        prev_idx = df.index[idx_loc - 1]
                        prev_close = df.loc[prev_idx, 'Close']
                        curr_open = df.loc[idx, 'Open']
                        df.loc[idx, 'Price_Change'] = curr_open - prev_close

                        # Extend the lagged position for this day (was 0, should be prev day's position)
                        prev_lagged_pos = df.iloc[idx_loc - 1]['Lagged_Position']
                        df.loc[idx, 'Lagged_Position'] = prev_lagged_pos

            # Calculate unslipped PnL
            df['Unslipped_PnL'] = df['Lagged_Position'] * df['Price_Change']
        else:
            # Default behavior: entry at close price
            # Unslipped PnL = lagged_position × (price_change)
            df['Price_Change'] = df['Close'].diff()
            df['Unslipped_PnL'] = df['Lagged_Position'] * df['Price_Change']

        # Fill NaN with 0 (first day)
        df['Unslipped_PnL'] = df['Unslipped_PnL'].fillna(0)

        logger.info("Calculated unslipped PnL")

        # Step 5: Calculate slippage (vectorized)
        # Slippage = |position_change| × price × slippage_bps / 10000
        # Note: slippage_bps / 10000 converts basis points to decimal
        # Example: 5 bps = 5/10000 = 0.0005 = 0.05%

        logger.info(f"Calculating slippage ({slippage_bps} bps)...")

        # Position change from previous day
        df['Position_Change'] = df['Exposure_Adjusted_Position'].diff().fillna(0)

        # Calculate slippage price (Close by default, Open for enter_at_open/exit_at_open)
        if enter_at_open or enter_at_next_close or exit_at_open:
            df['Slippage_Price'] = df['Close'].copy()

            if enter_at_open:
                # For enter_at_open: use Open price for entry slippage
                position_was_zero = df['Exposure_Adjusted_Position'].shift(1).fillna(0) == 0
                position_is_nonzero = df['Exposure_Adjusted_Position'] != 0
                is_entry = position_was_zero & position_is_nonzero
                df.loc[is_entry, 'Slippage_Price'] = df.loc[is_entry, 'Open']

            if enter_at_next_close:
                # For enter_at_next_close: slippage still occurs at Close price
                # but on the day AFTER the signal (when we actually enter)
                # The position change happens on signal day, so slippage uses Close price
                # No special handling needed - slippage naturally occurs at Close
                pass

            if exit_at_open:
                # For exit_at_open: use Open price for exit slippage
                actual_position_was_nonzero = df['Exposure_Adjusted_Position'].shift(1).fillna(0) != 0
                actual_position_is_zero = df['Exposure_Adjusted_Position'] == 0
                is_exit_day = actual_position_was_nonzero & actual_position_is_zero
                df.loc[is_exit_day, 'Slippage_Price'] = df.loc[is_exit_day, 'Open']

            df['Slippage'] = np.abs(df['Position_Change']) * df['Slippage_Price'] * (slippage_bps / 10000)
        else:
            # Default: use Close price for all slippage
            df['Slippage'] = np.abs(df['Position_Change']) * df['Close'] * (slippage_bps / 10000)

        # Step 6: Calculate slipped PnL (vectorized)
        df['Slipped_PnL'] = df['Unslipped_PnL'] - df['Slippage']

        logger.info("Calculated slipped PnL")

        # Step 7: Calculate cumulative PnL
        df['Cumulative_Unslipped_PnL'] = df['Unslipped_PnL'].cumsum()
        df['Cumulative_Slipped_PnL'] = df['Slipped_PnL'].cumsum()

        # Store the enhanced dataframe back
        self.strategy.df = df

        # Step 8: Calculate performance statistics
        logger.info("Calculating performance statistics...")

        # Unslipped performance
        self.unslipped_performance = stats(df['Unslipped_PnL'], df['Position'])
        logger.info(f"Unslipped - Sharpe: {self.unslipped_performance['sharpe']:.3f}, "
                   f"Total PnL: ${self.unslipped_performance['total_pnl']:,.2f}")

        # Slipped performance
        self.slipped_performance = stats(df['Slipped_PnL'], df['Position'])
        logger.info(f"Slipped - Sharpe: {self.slipped_performance['sharpe']:.3f}, "
                   f"Total PnL: ${self.slipped_performance['total_pnl']:,.2f}")

        # Calculate slippage cost
        total_slippage = df['Slippage'].sum()
        logger.info(f"Total slippage cost: ${total_slippage:,.2f}")

        logger.info(f"Backtest complete for {symbol}")

    def get_summary(self):
        """
        Get a formatted summary of backtest results.

        Returns:
            str: Formatted summary string
        """
        if self.slipped_performance is None or self.unslipped_performance is None:
            return "No backtest results available. Run backtest first."

        lines = []
        lines.append("=" * 80)
        lines.append(f"BACKTEST RESULTS: {self.symbol}")
        lines.append(f"Period: {self.start_date} to {self.end_date}")
        if self.exposure_manager is not None:
            lines.append(f"Exposure Manager: {self.exposure_manager}")
        else:
            lines.append(f"Dollar Size: ${self.dollar_size:,.2f}")
        lines.append(f"Slippage: {self.slippage_bps} bps")
        lines.append("=" * 80)

        lines.append("\nUNSLIPPED PERFORMANCE:")
        lines.append("-" * 80)
        unslipped = self.unslipped_performance
        lines.append(f"Total P&L:              ${unslipped['total_pnl']:,.2f}")
        lines.append(f"Sharpe Ratio:           {unslipped['sharpe']:.3f}")
        lines.append(f"Number of Trades:       {unslipped['num_trades']:,}")
        lines.append(f"Mean P&L per Trade:     ${unslipped['mean_pnl_per_trade']:,.2f}")

        lines.append("\nSLIPPED PERFORMANCE:")
        lines.append("-" * 80)
        slipped = self.slipped_performance
        lines.append(f"Total P&L:              ${slipped['total_pnl']:,.2f}")
        lines.append(f"Sharpe Ratio:           {slipped['sharpe']:.3f}")
        lines.append(f"Number of Trades:       {slipped['num_trades']:,}")
        lines.append(f"Mean P&L per Trade:     ${slipped['mean_pnl_per_trade']:,.2f}")

        lines.append("\nSLIPPAGE IMPACT:")
        lines.append("-" * 80)
        slippage_cost = unslipped['total_pnl'] - slipped['total_pnl']
        slippage_pct = (slippage_cost / abs(unslipped['total_pnl']) * 100) if unslipped['total_pnl'] != 0 else 0
        lines.append(f"Total Slippage Cost:    ${slippage_cost:,.2f} ({slippage_pct:.2f}%)")
        sharpe_impact = unslipped['sharpe'] - slipped['sharpe']
        lines.append(f"Sharpe Impact:          {sharpe_impact:.3f}")

        lines.append("=" * 80)

        return "\n".join(lines)

    def get_dataframe(self):
        """
        Get the strategy dataframe with all calculated fields.

        Returns:
            pd.DataFrame: DataFrame with prices, positions, PnL, etc.
        """
        if self.strategy.df is None:
            logger.warning("No dataframe available. Run backtest first.")
            return None

        return self.strategy.df

    def export_trades(self, output_path):
        """
        Export individual trade details to a CSV file.

        This method generates a CSV file with one row per trade containing entry/exit
        dates, prices, PnL, and days held.

        Args:
            output_path (str): Path where the CSV file will be saved
                              Example: 'output/AAPL-trades.csv'

        Returns:
            None

        Example:
            backtester('AAPL', '2024-01-01', '2024-12-31', slippage_bps=5)
            backtester.export_trades('output/AAPL-trades.csv')
        """
        if self.strategy.df is None or self.slipped_performance is None:
            logger.error("No backtest results available. Run backtest first.")
            return

        from .performance import extract_trades

        df = self.strategy.df

        # Check if strategy uses enter_at_open or exit_at_open
        use_open_entry = hasattr(self.strategy, '_enter_at_open') and self.strategy._enter_at_open
        use_open_exit = hasattr(self.strategy, '_exit_at_open') and self.strategy._exit_at_open

        # Extract trades
        trades_df = extract_trades(df, use_open_for_entry=use_open_entry, use_open_for_exit=use_open_exit)

        if trades_df.empty:
            logger.warning("No trades found to export.")
            print("WARNING: No trades found to export.")
            return

        # Save to CSV
        trades_df.to_csv(output_path, index=False)
        logger.info(f"Exported {len(trades_df)} trades to {output_path}")

    def visualize(self, output_path, bnh_backtester=None):
        """
        Create a visualization of cumulative PnL plots (slipped and unslipped).

        This method generates a PNG file containing:
        - Cumulative unslipped PnL curve (blue line)
        - Cumulative slipped PnL curve (red line)
        - Cumulative Buy & Hold PnL curve (green line, optional)
        - Legend showing Sharpe ratios for each

        Args:
            output_path (str): Path where the PNG file will be saved
                              Example: 'output/AAPL-dma.png'
            bnh_backtester (Backtester, optional): Buy & Hold backtester for comparison

        Returns:
            None

        Example:
            backtester('AAPL', '2024-01-01', '2024-12-31', slippage_bps=5)
            backtester.visualize('output/AAPL-dma.png')

            # With Buy & Hold comparison:
            backtester.visualize('output/AAPL-dma.png', bnh_backtester=bnh_bt)
        """
        if self.strategy.df is None or self.slipped_performance is None:
            logger.error("No backtest results available. Run backtest first.")
            return

        # Lazy import matplotlib (only when visualization is needed)
        try:
            import matplotlib
            matplotlib.use('Agg')  # Use non-interactive backend
            import matplotlib.pyplot as plt
        except ImportError as e:
            logger.error(f"matplotlib is required for visualization. Install with: pip install matplotlib ({e})")
            print("ERROR: matplotlib is not installed. Install with: pip install matplotlib")
            return
        except Exception as e:
            logger.warning(f"matplotlib warning (continuing): {e}")

        df = self.strategy.df

        # Create figure with conditional dual y-axes
        fig, ax1 = plt.subplots(figsize=(12, 6))

        # Check if we need dual y-axes (when benchmark is provided)
        use_dual_axes = bnh_backtester is not None and bnh_backtester.strategy.df is not None

        if use_dual_axes:
            # Plot strategy PnL on left y-axis
            ax1.plot(df.index, df['Cumulative_Unslipped_PnL'],
                    label=f'Unslipped (Sharpe: {self.unslipped_performance["sharpe"]:.3f})',
                    linewidth=1, color='blue', alpha=0.5)
            ax1.plot(df.index, df['Cumulative_Slipped_PnL'],
                    label=f'Slipped (Sharpe: {self.slipped_performance["sharpe"]:.3f})',
                    linewidth=3, color='red')

            # Set y-limits for left axis to use full scale
            strat_min = min(df['Cumulative_Unslipped_PnL'].min(), df['Cumulative_Slipped_PnL'].min())
            strat_max = max(df['Cumulative_Unslipped_PnL'].max(), df['Cumulative_Slipped_PnL'].max())
            strat_range = strat_max - strat_min
            ax1.set_ylim(strat_min - 0.05 * strat_range, strat_max + 0.05 * strat_range)

            # Create second y-axis for Buy & Hold
            ax2 = ax1.twinx()
            bnh_df = bnh_backtester.strategy.df
            bnh_sharpe = bnh_backtester.slipped_performance['sharpe']
            ax2.plot(bnh_df.index, bnh_df['Cumulative_Slipped_PnL'],
                    label=f'Buy & Hold (Sharpe: {bnh_sharpe:.3f})',
                    linewidth=1, color='green', alpha=0.5)

            # Set y-limits for right axis to use full scale
            bnh_min = bnh_df['Cumulative_Slipped_PnL'].min()
            bnh_max = bnh_df['Cumulative_Slipped_PnL'].max()
            bnh_range = bnh_max - bnh_min
            ax2.set_ylim(bnh_min - 0.05 * bnh_range, bnh_max + 0.05 * bnh_range)

            # Format both y-axes
            ax1.set_ylabel('Strategy Cumulative P&L ($)', fontsize=12, color='blue', fontweight='bold')
            ax2.set_ylabel('Buy & Hold Cumulative P&L ($)', fontsize=12, color='green', fontweight='bold')
            ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
            ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
            ax1.tick_params(axis='y', labelcolor='blue')
            ax2.tick_params(axis='y', labelcolor='green')

            # Combine legends from both axes
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='best', fontsize=10)
        else:
            # Single y-axis (no benchmark)
            ax1.plot(df.index, df['Cumulative_Unslipped_PnL'],
                    label=f'Unslipped (Sharpe: {self.unslipped_performance["sharpe"]:.3f})',
                    linewidth=2, color='blue')
            ax1.plot(df.index, df['Cumulative_Slipped_PnL'],
                    label=f'Slipped (Sharpe: {self.slipped_performance["sharpe"]:.3f})',
                    linewidth=2, color='red', linestyle='--')
            ax1.set_ylabel('Cumulative P&L ($)', fontsize=12)
            ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
            ax1.legend(loc='best', fontsize=10)

        # Add labels and title
        ax1.set_xlabel('Date', fontsize=12)

        # Update title based on whether comparison is shown
        if bnh_backtester is not None:
            title = f'{self.symbol} - Strategy Comparison (Dual Y-Axis)\n{self.start_date} to {self.end_date}'
        else:
            title = f'{self.symbol} - {self.strategy.__class__.__name__} Strategy\n{self.start_date} to {self.end_date}'

        ax1.set_title(title, fontsize=14, fontweight='bold')

        # Add grid
        ax1.grid(True, alpha=0.3)

        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45, ha='right')

        # Tight layout to prevent label cutoff
        plt.tight_layout()

        # Save figure
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Visualization saved to {output_path}")

        # Auto-export trades CSV with same prefix as PNG
        import os
        csv_path = output_path.rsplit('.', 1)[0] + '.csv'
        self.export_trades(csv_path)


if __name__ == "__main__":
    # Example usage
    import sys
    from pathlib import Path

    # Add parent directory to path
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from strategies.base import MovingAverageCrossoverStrategy, BuyAndHoldStrategy
    from functools import partial
    from libs.volatility import simple_vol
    import logging

    logging.basicConfig(level=logging.INFO)

    print("=" * 80)
    print("Backtester Examples (Exposure-Adjusted with Slippage)")
    print("=" * 80)

    # Example 1: Single backtest with slippage
    print("\nExample 1: MA Strategy with 5 bps slippage")
    print("-" * 80)

    strategy = MovingAverageCrossoverStrategy(
        volatility_function=partial(simple_vol, N=20),
        short_period=20,
        long_period=50
    )
    backtester = Backtester(strategy, dollar_size=100000)

    backtester('AAPL', '2024-01-01', '2024-12-31', slippage_bps=5)

    print(backtester.get_summary())

    # Example 2: Compare different slippage levels
    print("\nExample 2: Slippage Sensitivity Analysis")
    print("-" * 80)

    slippage_levels = [0, 2, 5, 10, 20]
    results = []

    for slippage in slippage_levels:
        # Create fresh strategy instance
        strat = MovingAverageCrossoverStrategy(short_period=20, long_period=50)
        bt = Backtester(strat, dollar_size=100000)

        bt('AAPL', '2024-01-01', '2024-12-31', slippage_bps=slippage)

        results.append({
            'Slippage (bps)': slippage,
            'Unslipped PnL': bt.unslipped_performance['total_pnl'],
            'Slipped PnL': bt.slipped_performance['total_pnl'],
            'Cost': bt.unslipped_performance['total_pnl'] - bt.slipped_performance['total_pnl'],
            'Unslipped Sharpe': bt.unslipped_performance['sharpe'],
            'Slipped Sharpe': bt.slipped_performance['sharpe']
        })

    import pandas as pd
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))

    # Example 3: View detailed data
    print("\nExample 3: Sample Detailed Data (Last 10 Days)")
    print("-" * 80)

    df = backtester.get_dataframe()
    if df is not None:
        cols = ['Close', 'Volatility', 'Dollar_Volatility', 'Position',
                'Exposure_Adjusted_Position', 'Unslipped_PnL', 'Slippage', 'Slipped_PnL']
        print(df[cols].tail(10).to_string())

    # Example 4: Different dollar sizes
    print("\nExample 4: Different Dollar Sizes")
    print("-" * 80)

    dollar_sizes = [50000, 100000, 250000, 500000]
    size_results = []

    for size in dollar_sizes:
        strat = MovingAverageCrossoverStrategy(short_period=20, long_period=50)
        bt = Backtester(strat, dollar_size=size)

        bt('AAPL', '2024-01-01', '2024-12-31', slippage_bps=5)

        size_results.append({
            'Dollar Size': f'${size:,}',
            'Total PnL': bt.slipped_performance['total_pnl'],
            'Sharpe': bt.slipped_performance['sharpe'],
            'Num Trades': bt.slipped_performance['num_trades']
        })

    size_df = pd.DataFrame(size_results)
    print(size_df.to_string(index=False))

    print("\n" + "=" * 80)
    print("Examples completed successfully!")
    print("\nKey Features Demonstrated:")
    print("- Exposure-adjusted position sizing")
    print("- Slippage modeling (basis points)")
    print("- Unslipped vs Slipped performance comparison")
    print("- Fully vectorized calculations")
    print("=" * 80)
