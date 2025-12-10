"""
Portfolio Backtester Module

This module provides the PortfolioBacktester class to run trading strategies
across multiple symbols and aggregate performance.
"""

import pandas as pd
import numpy as np
import logging
from .backtester import Backtester


logger = logging.getLogger(__name__)


class PortfolioBacktester:
    """
    Portfolio Backtester class to run trading strategies across multiple symbols.

    The PortfolioBacktester:
    1. Takes a list of symbols
    2. Runs the backtester on each symbol
    3. Stores all results
    4. Aggregates performance across the portfolio
    5. Creates visualizations of cumulative PnL curves
    """

    def __init__(self, strategy_class, strategy_params=None, dollar_size=100000):
        """
        Initialize the Portfolio Backtester.

        Args:
            strategy_class: Strategy class (not instance) to use for all symbols
            strategy_params (dict): Parameters to pass to strategy constructor
            dollar_size (float): Target dollar size for position sizing per symbol

        Example:
            from strategies.dma import DMA
            from libs.portfolio import PortfolioBacktester

            portfolio = PortfolioBacktester(
                strategy_class=DMA,
                strategy_params={'lookback': 200},
                dollar_size=100000
            )
        """
        self.strategy_class = strategy_class
        self.strategy_params = strategy_params or {}
        self.dollar_size = dollar_size

        # Results storage
        self.symbols = []
        self.backtesters = {}
        self.start_date = None
        self.end_date = None
        self.slippage_bps = None

        # Aggregated results
        self.portfolio_performance = None
        self.portfolio_df = None

        logger.info(f"Initialized PortfolioBacktester with {strategy_class.__name__}")
        logger.info(f"Dollar size per symbol: ${dollar_size:,.2f}")

    def __call__(self, symbols, start_date, end_date, slippage_bps=0):
        """
        Run the strategy on a list of symbols and aggregate results.

        Args:
            symbols (list): List of stock ticker symbols
            start_date (str|datetime): Start date (YYYY-MM-DD)
            end_date (str|datetime): End date (YYYY-MM-DD)
            slippage_bps (float): Slippage in basis points (default: 0)

        Returns:
            None (results stored in self attributes)

        Example:
            portfolio(['AAPL', 'MSFT', 'GOOGL'], '2024-01-01', '2024-12-31', slippage_bps=5)
            print(portfolio.get_summary())
        """
        self.symbols = symbols
        self.start_date = start_date
        self.end_date = end_date
        self.slippage_bps = slippage_bps

        logger.info(f"Running portfolio backtest on {len(symbols)} symbols")
        logger.info(f"Period: {start_date} to {end_date}")
        logger.info(f"Slippage: {slippage_bps} bps")

        # Run backtest for each symbol
        successful_symbols = []
        failed_symbols = []

        for symbol in symbols:
            logger.info(f"Processing {symbol}...")

            try:
                # Create fresh strategy instance for this symbol
                strategy = self.strategy_class(**self.strategy_params)

                # Create backtester
                backtester = Backtester(strategy, dollar_size=self.dollar_size)

                # Run backtest
                backtester(symbol, start_date, end_date, slippage_bps)

                # Check if backtest was successful
                if backtester.slipped_performance is not None:
                    self.backtesters[symbol] = backtester
                    successful_symbols.append(symbol)
                    logger.info(f"{symbol} completed successfully")
                else:
                    failed_symbols.append(symbol)
                    logger.warning(f"{symbol} backtest failed")

            except Exception as e:
                failed_symbols.append(symbol)
                logger.error(f"Error processing {symbol}: {str(e)}")

        # Store successful symbols
        self.symbols = successful_symbols

        logger.info(f"Portfolio backtest complete: {len(successful_symbols)} successful, {len(failed_symbols)} failed")

        if failed_symbols:
            logger.warning(f"Failed symbols: {', '.join(failed_symbols)}")

        # Aggregate performance
        if successful_symbols:
            self._aggregate_performance()
        else:
            logger.error("No successful backtests. Cannot aggregate performance.")
            self.portfolio_performance = None

    def _aggregate_performance(self):
        """
        Aggregate performance across all symbols in the portfolio.

        This creates a portfolio-level PnL series by summing the PnL
        from all symbols and calculating portfolio-level statistics.
        """
        logger.info("Aggregating portfolio performance...")

        # Collect all dataframes and position data
        dfs = []
        all_positions = []
        for symbol in self.symbols:
            backtester = self.backtesters[symbol]
            df = backtester.get_dataframe()
            if df is not None:
                # Extract relevant columns and rename with symbol prefix
                df_subset = df[['Slipped_PnL', 'Cumulative_Slipped_PnL']].copy()
                df_subset.columns = [f'{symbol}_PnL', f'{symbol}_Cumulative_PnL']
                dfs.append(df_subset)
                
                # Collect position data for exposure calculation
                # Look for position column (could be 'Position' or 'Exposure_Adjusted_Position')
                pos_col = None
                for col in ['Position', 'Exposure_Adjusted_Position']:
                    if col in df.columns:
                        pos_col = col
                        break
                
                if pos_col:
                    # Get position values as a Series with the same index
                    pos_series = df[pos_col].copy()
                else:
                    # If no position column found, create a series of zeros
                    pos_series = pd.Series(0, index=df.index)
                
                all_positions.append(pos_series)

        if not dfs:
            logger.error("No valid dataframes to aggregate")
            self.portfolio_performance = None
            return

        # Merge all dataframes on date index (outer join to handle missing dates)
        portfolio_df = pd.concat(dfs, axis=1, join='outer')

        # Fill NaN values with 0 (for days when a symbol wasn't trading)
        portfolio_df = portfolio_df.fillna(0)

        # Calculate portfolio-level PnL
        pnl_cols = [col for col in portfolio_df.columns if col.endswith('_PnL') and not col.endswith('Cumulative_PnL')]
        portfolio_df['Portfolio_PnL'] = portfolio_df[pnl_cols].sum(axis=1)
        portfolio_df['Portfolio_Cumulative_PnL'] = portfolio_df['Portfolio_PnL'].cumsum()

        # Store portfolio dataframe
        self.portfolio_df = portfolio_df

        # Calculate portfolio statistics using all days
        from .performance import stats
        self.portfolio_performance = stats(portfolio_df['Portfolio_PnL'])
        
        # Calculate exposure-based metrics
        # Align all position series to the same index (portfolio_df index)
        aligned_positions = []
        for pos_series in all_positions:
            # Reindex to portfolio_df index, filling missing values with 0
            aligned_pos = pos_series.reindex(portfolio_df.index, fill_value=0)
            aligned_positions.append(aligned_pos)
        
        # Create a combined exposure mask: True if any symbol has a non-zero position
        if aligned_positions:
            # Convert to DataFrame for easier manipulation
            pos_df = pd.concat(aligned_positions, axis=1)
            # Create exposure mask (any non-zero position)
            exposure_mask = (pos_df.abs() > 1e-10).any(axis=1)
            num_exposure_days = exposure_mask.sum()
            
            # Calculate exposure-based Sharpe using only days with exposure
            if num_exposure_days > 0:
                # Get portfolio PnL on exposure days
                exposure_pnl = portfolio_df['Portfolio_PnL'][exposure_mask]
                # Calculate daily returns
                exposure_daily_returns = exposure_pnl / self.dollar_size
                # Calculate Sharpe ratio
                if len(exposure_daily_returns) > 1 and exposure_daily_returns.std() > 0:
                    sharpe_exposure = np.sqrt(252) * exposure_daily_returns.mean() / exposure_daily_returns.std()
                else:
                    sharpe_exposure = 0
            else:
                sharpe_exposure = 0
                num_exposure_days = 0
        else:
            sharpe_exposure = 0
            num_exposure_days = 0
        
        # Add exposure-based metrics to portfolio performance
        self.portfolio_performance['sharpe_exposure'] = sharpe_exposure
        self.portfolio_performance['num_exposure_days'] = num_exposure_days
        self.portfolio_performance['pnl_per_exposure_day'] = (
            self.portfolio_performance['total_pnl'] / num_exposure_days 
            if num_exposure_days > 0 else 0
        )

        logger.info(f"Portfolio Sharpe: {self.portfolio_performance['sharpe']:.3f}")
        logger.info(f"Portfolio Sharpe (Exposure): {self.portfolio_performance['sharpe_exposure']:.3f}")
        logger.info(f"Portfolio Total PnL: ${self.portfolio_performance['total_pnl']:,.2f}")
        logger.info(f"Number of exposure days: {num_exposure_days}")

    def get_summary(self):
        """
        Get a formatted summary of portfolio backtest results.

        Returns:
            str: Formatted summary string
        """
        if self.portfolio_performance is None:
            return "No portfolio results available. Run portfolio backtest first."

        lines = []
        lines.append("=" * 80)
        lines.append("PORTFOLIO BACKTEST RESULTS")
        lines.append(f"Symbols: {', '.join(self.symbols)}")
        lines.append(f"Period: {self.start_date} to {self.end_date}")
        lines.append(f"Dollar Size per Symbol: ${self.dollar_size:,.2f}")
        lines.append(f"Slippage: {self.slippage_bps} bps")
        lines.append("=" * 80)

        # Individual symbol performance
        lines.append("\nINDIVIDUAL SYMBOL PERFORMANCE:")
        lines.append("-" * 80)
        lines.append(f"{'Symbol':<10} {'Total PnL':>15} {'Sharpe':>10} {'#Trades':>10}")
        lines.append("-" * 80)

        for symbol in self.symbols:
            bt = self.backtesters[symbol]
            perf = bt.slipped_performance
            lines.append(f"{symbol:<10} ${perf['total_pnl']:>14,.2f} {perf['sharpe']:>10.3f} {perf['num_trades']:>10,}")

        # Portfolio aggregated performance
        lines.append("\nPORTFOLIO AGGREGATED PERFORMANCE:")
        lines.append("-" * 80)
        port = self.portfolio_performance
        lines.append(f"Total P&L:              ${port['total_pnl']:,.2f}")
        lines.append(f"Sharpe Ratio:           {port['sharpe']:.3f}")
        lines.append(f"Number of Trades:       {port['num_trades']:,}")
        lines.append(f"Mean P&L per Trade:     ${port['mean_pnl_per_trade']:,.2f}")
        lines.append(f"Number of Wins:         {port['num_wins']:,}")
        lines.append(f"Average P&L per Win:    ${port['avg_pnl_win']:,.2f}")
        lines.append(f"Days Held (Wins):       {port['days_held_wins']:.1f}")
        lines.append(f"Number of Losses:       {port['num_losses']:,}")
        lines.append(f"Average P&L per Loss:   ${port['avg_pnl_loss']:,.2f}")
        lines.append(f"Days Held (Losses):     {port['days_held_losses']:.1f}")
        lines.append(f"Max Drawdown:           ${port['max_drawdown']:,.2f}")

        lines.append("=" * 80)

        return "\n".join(lines)

    def visualize(self, output_path):
        """
        Create a visualization of portfolio cumulative PnL curves.

        This method generates a PNG file containing:
        - Individual symbol cumulative PnL curves
        - Portfolio aggregated cumulative PnL curve
        - Legend showing Sharpe ratios

        Args:
            output_path (str): Path where the PNG file will be saved
                              Example: 'output/portfolio-dma.png'

        Returns:
            None

        Example:
            portfolio(['AAPL', 'MSFT'], '2024-01-01', '2024-12-31', slippage_bps=5)
            portfolio.visualize('output/portfolio-dma.png')
        """
        if self.portfolio_df is None or self.portfolio_performance is None:
            logger.error("No portfolio results available. Run portfolio backtest first.")
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

        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

        # Subplot 1: Individual symbols
        for symbol in self.symbols:
            bt = self.backtesters[symbol]
            df = bt.get_dataframe()
            sharpe = bt.slipped_performance['sharpe']
            ax1.plot(df.index, df['Cumulative_Slipped_PnL'],
                    label=f'{symbol} (Sharpe: {sharpe:.3f})',
                    linewidth=1.5, alpha=0.7)

        ax1.set_xlabel('Date', fontsize=11)
        ax1.set_ylabel('Cumulative P&L ($)', fontsize=11)
        ax1.set_title('Individual Symbol Performance',
                     fontsize=13, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='best', fontsize=9)
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

        # Subplot 2: Portfolio aggregate
        portfolio_sharpe = self.portfolio_performance['sharpe']
        ax2.plot(self.portfolio_df.index, self.portfolio_df['Portfolio_Cumulative_PnL'],
                label=f'Portfolio (Sharpe: {portfolio_sharpe:.3f})',
                linewidth=2.5, color='darkblue')

        ax2.set_xlabel('Date', fontsize=11)
        ax2.set_ylabel('Cumulative P&L ($)', fontsize=11)
        ax2.set_title('Portfolio Aggregated Performance',
                     fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='best', fontsize=10)
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

        # Main title
        fig.suptitle(f'Portfolio Backtest - {self.strategy_class.__name__}\n'
                    f'{", ".join(self.symbols)}\n'
                    f'{self.start_date} to {self.end_date}',
                    fontsize=14, fontweight='bold', y=0.995)

        # Rotate x-axis labels
        for ax in [ax1, ax2]:
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # Tight layout
        plt.tight_layout()

        # Save figure
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Portfolio visualization saved to {output_path}")

    def get_dataframe(self):
        """
        Get the portfolio dataframe with all PnL series.

        Returns:
            pd.DataFrame: DataFrame with individual symbol and portfolio PnL
        """
        if self.portfolio_df is None:
            logger.warning("No portfolio dataframe available. Run portfolio backtest first.")
            return None

        return self.portfolio_df


if __name__ == "__main__":
    # Example usage
    import sys
    from pathlib import Path

    # Add parent directory to path
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from strategies.dma import DMA
    import logging

    logging.basicConfig(level=logging.INFO)

    print("=" * 80)
    print("Portfolio Backtester Example")
    print("=" * 80)

    # Create portfolio backtester
    portfolio = PortfolioBacktester(
        strategy_class=DMA,
        strategy_params={'lookback': 200},
        dollar_size=100000
    )

    # Run on multiple symbols
    symbols = ['AAPL', 'MSFT', 'GOOGL']
    portfolio(symbols, '2024-01-01', '2024-12-31', slippage_bps=5)

    # Display results
    print()
    print(portfolio.get_summary())

    # Save visualization
    portfolio.visualize('output/portfolio-200dma.png')

    print()
    print("=" * 80)
    print("Portfolio backtest completed successfully!")
    print("=" * 80)
