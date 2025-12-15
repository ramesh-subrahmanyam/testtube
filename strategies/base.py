"""
Base Strategy Class for Backtesting

This module provides an abstract base class for implementing trading strategies.
All concrete strategies should inherit from BaseStrategy and implement the
required abstract methods.
"""

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
import logging
from datetime import datetime

import sys
from pathlib import Path

# Add parent directory to path to import libs
sys.path.insert(0, str(Path(__file__).parent.parent))

from libs.prices import fetch_price_range
from libs.volatility import simple_vol
from functools import partial


logger = logging.getLogger(__name__)


class BaseStrategy(ABC):
    """
    Abstract base class for trading strategies.

    A strategy determines daily positions (long, short, or flat) based on
    price data and strategy-specific parameters.

    The strategy also calculates rolling volatility using a configurable
    volatility function.

    Subclasses must implement:
    - generate_signals(): Logic to generate buy/sell/hold signals
    """

    def __init__(self, volatility_function=None, **strategy_params):
        """
        Initialize the strategy.

        Args:
            volatility_function (callable): Function to calculate volatility.
                                           Should take prices and return volatility series.
                                           Default: simple_vol with N=20
            **strategy_params: Strategy-specific parameters (e.g., ma_period=50)

        Example:
            from libs.volatility import simple_vol
            from functools import partial

            # Use 30-day simple volatility
            strategy = MyStrategy(volatility_function=partial(simple_vol, N=30),
                                 short_ma=20, long_ma=50)
        """
        # Default volatility function: 20-day simple volatility
        if volatility_function is None:
            self.volatility_function = partial(simple_vol, N=20)
        else:
            self.volatility_function = volatility_function

        self.strategy_params = strategy_params

        # Will be set when __call__ is invoked
        self.symbol = None
        self.start_date = None
        self.end_date = None
        self.prices = None
        self.df = None  # DataFrame with prices, volatility, and positions

        logger.info(f"Initialized {self.__class__.__name__}")
        logger.info(f"Volatility function: {self.volatility_function}")
        logger.info(f"Strategy parameters: {strategy_params}")

    def __call__(self, symbol, start_date, end_date):
        """
        Run the strategy for a given symbol and date range.

        This method:
        1. Fetches price data using libs/prices.py
        2. Calculates rolling volatility using the volatility function
        3. Calls generate_signals() to create trading signals
        4. Stores prices, volatility, and positions in self.df
        5. Returns self.df

        Args:
            symbol (str): Stock ticker symbol
            start_date (str|datetime): Start date for backtesting (YYYY-MM-DD)
            end_date (str|datetime): End date for backtesting (YYYY-MM-DD)

        Returns:
            pd.DataFrame: self.df with columns:
                - Date (index): Trading date
                - OHLCV columns (Open, High, Low, Close, Volume)
                - Volatility: Rolling volatility
                - Signal: Trading signal (1=long, 0=flat, -1=short)
                - Position: Actual position (may differ from signal due to execution rules)
                Plus any additional columns added by generate_signals()
        """
        self.symbol = symbol
        self.start_date = start_date if isinstance(start_date, str) else start_date.strftime('%Y-%m-%d')
        self.end_date = end_date if isinstance(end_date, str) else end_date.strftime('%Y-%m-%d')

        logger.info(f"Running {self.__class__.__name__} on {symbol} from {self.start_date} to {self.end_date}")

        # Fetch price data
        self.prices = self._fetch_prices(symbol)

        if self.prices is None or self.prices.empty:
            logger.error(f"No price data available for {symbol}")
            return None

        logger.info(f"Fetched {len(self.prices)} days of data for {symbol}")

        # Calculate volatility
        logger.info(f"Calculating volatility using {self.volatility_function}")
        volatility = self.volatility_function(self.prices)

        # Add volatility to prices
        self.prices['Volatility'] = volatility

        # Generate signals using strategy-specific logic
        signals_df = self.generate_signals(self.prices)

        if signals_df is None:
            logger.error(f"Signal generation failed for {symbol}")
            return None

        # Validate signals
        if not self._validate_signals(signals_df):
            logger.error(f"Signal validation failed for {symbol}")
            return None

        # Trim initial NaN values from warm-up period
        signals_df = self._trim_warmup_period(signals_df)

        # Store complete DataFrame with prices, volatility, and positions
        self.df = signals_df

        return self.df

    def _fetch_prices(self, symbol):
        """
        Fetch price data for the strategy period.

        Uses libs/prices.py with caching enabled by default.

        Args:
            symbol (str): Stock ticker

        Returns:
            pd.DataFrame: OHLCV data with DatetimeIndex
        """
        try:
            # Fetch with some buffer before start_date for indicator calculation
            # Subtract 100 days to ensure we have enough data for moving averages, etc.
            buffer_start = pd.to_datetime(self.start_date) - pd.Timedelta(days=100)
            buffer_start_str = buffer_start.strftime('%Y-%m-%d')

            data = fetch_price_range(symbol, buffer_start_str, self.end_date,
                                    use_cache=True, force_refresh=False)

            return data

        except Exception as e:
            logger.error(f"Error fetching prices for {symbol}: {e}")
            return None

    @abstractmethod
    def generate_signals(self, prices):
        """
        Generate trading signals based on price data.

        This is the core strategy logic that must be implemented by subclasses.

        NEW SCHEMA: Strategies must now return:
        - Signal: Real-valued signal (can be any float, not just -1/0/1)
        - Target_Position: Discrete target position (typically -1/0/1), set ONLY on signal days
        - Position_At_Close: Actual position at close (calculated from Target_Position)
        - Entry_Time: 'close' (default), will be modified by modifiers
        - Entry_Price: Price at which entry occurs (Close by default)

        Args:
            prices (pd.DataFrame): OHLCV data with DatetimeIndex

        Returns:
            pd.DataFrame: DataFrame with at minimum:
                - Date (index): Trading date
                - OHLCV columns
                - Signal: Real-valued signal
                - Target_Position: Discrete position (set only on signal days, NaN otherwise)
                - Position_At_Close: Actual position at close of each day
                - Entry_Time: Entry timing ('close', 'next_open', etc.)
                - Entry_Price: Entry price

        Example implementation:
            def generate_signals(self, prices):
                df = prices.copy()
                df['MA20'] = df['Close'].rolling(20).mean()
                df['MA50'] = df['Close'].rolling(50).mean()

                # Generate real-valued signal
                df['Signal'] = (df['MA20'] - df['MA50']) / df['MA50']

                # Convert to discrete target positions (set only when signal changes)
                df['Target_Position'] = np.nan
                df.loc[df['Signal'] > 0, 'Target_Position'] = 1
                df.loc[df['Signal'] < 0, 'Target_Position'] = -1
                df.loc[df['Signal'] == 0, 'Target_Position'] = 0

                # Forward-fill to get position at close
                df['Position_At_Close'] = df['Target_Position'].fillna(method='ffill').fillna(0)

                # Default entry timing
                df['Entry_Time'] = 'close'
                df['Entry_Price'] = df['Close']

                return df.loc[self.start_date:self.end_date]
        """
        pass

    def _trim_warmup_period(self, signals):
        """
        Trim initial NaN values from the warm-up period in the signal series.

        Args:
            signals (pd.DataFrame): Signals DataFrame

        Returns:
            pd.DataFrame: Signals with initial NaN rows removed
        """
        # Find the first non-NaN position in the Signal column
        first_valid_idx = signals['Signal'].first_valid_index()

        if first_valid_idx is None:
            # All NaNs - return empty DataFrame
            logger.warning("All signals are NaN - returning empty DataFrame")
            return signals.iloc[0:0]

        # Trim to start from first valid signal
        trimmed = signals.loc[first_valid_idx:]

        initial_rows_trimmed = len(signals) - len(trimmed)
        if initial_rows_trimmed > 0:
            logger.info(f"Trimmed {initial_rows_trimmed} rows from warm-up period")
            logger.info(f"Strategy period now starts from {first_valid_idx}")

        return trimmed

    def _validate_signals(self, signals):
        """
        Validate that signals DataFrame has required columns and handle NaNs.

        This method:
        1. Checks for required columns (new schema)
        2. Validates signal values
        3. Handles NaNs in the Signal column:
           - Excludes initial warm-up period NaNs
           - Fails if NaNs exist in the middle of the signal series
           - Creates an error file documenting mid-series NaNs

        Args:
            signals (pd.DataFrame): Signals to validate

        Returns:
            bool: True if valid, False otherwise
        """
        required_columns = ['Close', 'Signal', 'Target_Position', 'Position_At_Close', 'Entry_Time', 'Entry_Price']

        for col in required_columns:
            if col not in signals.columns:
                logger.error(f"Missing required column: {col}")
                return False

        # Check for NaNs in the Signal column
        signal_nans = signals['Signal'].isna()

        if signal_nans.any():
            # Find the first non-NaN position
            first_valid_idx = signals['Signal'].first_valid_index()

            if first_valid_idx is None:
                logger.error("Signal column contains all NaNs")
                return False

            # Check if there are NaNs after the first valid signal (mid-series NaNs)
            signals_after_warmup = signals.loc[first_valid_idx:]
            mid_series_nans = signals_after_warmup['Signal'].isna()

            if mid_series_nans.any():
                # Count and get dates of mid-series NaNs
                nan_count = mid_series_nans.sum()
                nan_dates = signals_after_warmup[mid_series_nans].index.tolist()

                # Create error file
                self._create_nan_error_file(nan_count, nan_dates)

                logger.error(f"Found {nan_count} NaN values in the middle of the signal series")
                return False

            # If we get here, NaNs are only at the beginning (warm-up period)
            # This is acceptable, but log it
            warmup_nan_count = signal_nans.sum()
            logger.info(f"Excluding {warmup_nan_count} initial NaN values from warm-up period")

        # Note: Signal is now real-valued, so we don't check for [-1, 0, 1]
        # Just ensure it's numeric
        if not pd.api.types.is_numeric_dtype(signals['Signal']):
            logger.error("Signal column must be numeric")
            return False

        # Check Position_At_Close values are valid (excluding NaNs)
        # Allow any numeric value (not restricted to -1, 0, 1 anymore)
        if not pd.api.types.is_numeric_dtype(signals['Position_At_Close']):
            logger.error("Position_At_Close column must be numeric")
            return False

        return True

    def _create_nan_error_file(self, nan_count, nan_dates):
        """
        Create an error file documenting NaN values in the middle of the signal series.

        Args:
            nan_count (int): Number of NaN values found
            nan_dates (list): List of dates where NaNs were found
        """
        import os
        from datetime import datetime as dt

        # Get the signal function name from the strategy class
        signal_function_name = f"{self.__class__.__name__}.generate_signals"

        # Create error filename with timestamp
        timestamp = dt.now().strftime('%Y%m%d_%H%M%S')
        error_filename = f"nan_error_{self.symbol}_{timestamp}.txt"

        # Write error file to current directory
        with open(error_filename, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("NaN VALUES DETECTED IN SIGNAL SERIES\n")
            f.write("=" * 70 + "\n\n")

            f.write(f"Strategy: {self.__class__.__name__}\n")
            f.write(f"Signal Function: {signal_function_name}\n")
            f.write(f"Symbol: {self.symbol}\n")
            f.write(f"Period: {self.start_date} to {self.end_date}\n")
            f.write(f"\nError Timestamp: {dt.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("\n" + "-" * 70 + "\n\n")

            f.write(f"ISSUE: NaN values found in the MIDDLE of the signal series\n")
            f.write(f"       (after the initial warm-up period)\n\n")

            f.write(f"Total NaN count: {nan_count}\n\n")

            f.write("Dates with NaN values:\n")
            f.write("-" * 70 + "\n")
            for i, date in enumerate(nan_dates, 1):
                date_str = date.strftime('%Y-%m-%d') if hasattr(date, 'strftime') else str(date)
                f.write(f"{i:4d}. {date_str}\n")

            f.write("\n" + "=" * 70 + "\n")
            f.write("RECOMMENDATION: Check the signal calculation logic to ensure\n")
            f.write("                all required data is available for these dates.\n")
            f.write("=" * 70 + "\n")

        logger.error(f"NaN error details written to: {error_filename}")
        print(f"\n⚠️  ERROR: NaN values detected in signal series!")
        print(f"    Error file created: {error_filename}\n")

    # =============================================================================
    # Strategy Modifier Methods (Fluent API)
    # =============================================================================

    def enter_at_next_open(self):
        """
        Wrap strategy to enter positions at next day's open price.

        Returns a EnterAtNextOpen modifier wrapping this strategy.

        Returns:
            EnterAtNextOpen: Wrapped strategy

        Example:
            strategy = DMA(200).enter_at_next_open()
        """
        from strategies.modifiers import EnterAtNextOpen
        return EnterAtNextOpen(self)

    def enter_at_next_close(self):
        """
        Wrap strategy to enter positions at next day's close price.

        Returns a EnterAtNextClose modifier wrapping this strategy.

        Returns:
            EnterAtNextClose: Wrapped strategy

        Example:
            strategy = DMA(200).enter_at_next_close()
        """
        from strategies.modifiers import EnterAtNextClose
        return EnterAtNextClose(self)

    def exit_at_next_open(self):
        """
        Wrap strategy to exit positions at next day's open price.

        Returns a ExitAtNextOpen modifier wrapping this strategy.

        Returns:
            ExitAtNextOpen: Wrapped strategy

        Example:
            strategy = DMA(200).exit_at_next_open()
        """
        from strategies.modifiers import ExitAtNextOpen
        return ExitAtNextOpen(self)

    def cap_position(self, max_position):
        """
        Wrap strategy to cap position sizes at a maximum value.

        Args:
            max_position (float): Maximum absolute position size

        Returns:
            CapPosition: Wrapped strategy

        Example:
            strategy = DMA(200).cap_position(1.0)
        """
        from strategies.modifiers import CapPosition
        return CapPosition(self, max_position)

    def only_on_days(self, allowed_days):
        """
        Wrap strategy to only trade on specific days of the week.

        Args:
            allowed_days (list): List of day names (e.g., ['Monday', 'Friday'])

        Returns:
            TradeOnlyOnDays: Wrapped strategy

        Example:
            strategy = DMA(200).only_on_days(['Monday', 'Wednesday', 'Friday'])
        """
        from strategies.modifiers import TradeOnlyOnDays
        return TradeOnlyOnDays(self, allowed_days)

    def minimum_signal_strength(self, min_strength):
        """
        Wrap strategy to filter out weak signals.

        Args:
            min_strength (float): Minimum absolute signal strength

        Returns:
            MinimumSignalStrength: Wrapped strategy

        Example:
            strategy = MyStrategy().minimum_signal_strength(0.1)
        """
        from strategies.modifiers import MinimumSignalStrength
        return MinimumSignalStrength(self, min_strength)

    def with_stop_loss(self, pct):
        """
        Wrap strategy with a stop loss.

        Args:
            pct (float): Stop loss percentage (e.g., 2.0 = 2%)

        Returns:
            WithStopLoss: Wrapped strategy

        Example:
            strategy = DMA(200).with_stop_loss(pct=2.0)
        """
        from strategies.modifiers import WithStopLoss
        return WithStopLoss(self, pct)

    def with_take_profit(self, pct):
        """
        Wrap strategy with a take profit target.

        Args:
            pct (float): Take profit percentage (e.g., 5.0 = 5%)

        Returns:
            WithTakeProfit: Wrapped strategy

        Example:
            strategy = DMA(200).with_take_profit(pct=5.0)
        """
        from strategies.modifiers import WithTakeProfit
        return WithTakeProfit(self, pct)

    def limit_order(self, limit_pct):
        """
        Wrap strategy to use limit orders instead of market orders.

        Args:
            limit_pct (float): Limit percentage (e.g., 0.5 = 0.5% below for long)

        Returns:
            LimitOrder: Wrapped strategy

        Example:
            strategy = DMA(200).limit_order(limit_pct=0.5)
        """
        from strategies.modifiers import LimitOrder
        return LimitOrder(self, limit_pct)

    def periodic(self, frequency='daily', when='end', day=None):
        """
        Wrap this strategy with periodic signal updates.

        This causes the strategy to only update its positions on specific dates
        (e.g., monthly rebalancing) rather than daily.

        Args:
            frequency (str): Update frequency - "daily", "weekly", "monthly"
            when (str): Update at period "start" or "end"
            day (str, optional): For weekly - day of week

        Returns:
            PeriodicStrategy: Wrapped strategy with periodic updates

        Example:
            # Monthly rebalancing at start of month
            strategy = MyStrategy().periodic(frequency="monthly", when="start")

            # Weekly rebalancing on Mondays
            strategy = MyStrategy().periodic(frequency="weekly", day="monday")
        """
        from libs.periodic import PeriodicStrategy
        return PeriodicStrategy(self, frequency=frequency, when=when, day=day)

    def get_parameter_summary(self):
        """
        Get a summary of strategy parameters.

        Returns:
            dict: Dictionary of strategy parameters
        """
        return {
            'strategy_name': self.__class__.__name__,
            'start_date': self.start_date,
            'end_date': self.end_date,
            **self.strategy_params
        }


class BuyAndHoldStrategy(BaseStrategy):
    """
    Simple buy-and-hold strategy (useful for benchmarking).

    Always holds a long position throughout the backtest period.
    """

    def generate_signals(self, prices):
        """Generate signals for buy-and-hold strategy."""
        df = prices.copy()

        # Always long - real-valued signal
        df['Signal'] = 1.0

        # Target position: always 1 (long) - set on first day only
        df['Target_Position'] = np.nan
        df.iloc[0, df.columns.get_loc('Target_Position')] = 1

        # Position at close: always 1
        df['Position_At_Close'] = 1

        # Default entry timing
        df['Entry_Time'] = 'close'
        df['Entry_Price'] = df['Close']

        # Return only the strategy period
        return df.loc[self.start_date:self.end_date]


class MovingAverageCrossoverStrategy(BaseStrategy):
    """
    Example: Moving Average Crossover Strategy

    Goes long when short MA > long MA, short when short MA < long MA.

    Parameters:
        short_period (int): Short moving average period (default 20)
        long_period (int): Long moving average period (default 50)
    """

    def __init__(self, short_period=20, long_period=50, **kwargs):
        super().__init__(short_period=short_period,
                        long_period=long_period,
                        **kwargs)
        self.short_period = short_period
        self.long_period = long_period

    def generate_signals(self, prices):
        """Generate signals based on moving average crossover."""
        df = prices.copy()

        # Calculate moving averages
        df['MA_Short'] = df['Close'].rolling(window=self.short_period).mean()
        df['MA_Long'] = df['Close'].rolling(window=self.long_period).mean()

        # Generate real-valued signal
        df['Signal'] = (df['MA_Short'] - df['MA_Long']) / df['MA_Long']

        # Convert to discrete positions first
        df['_temp_position'] = 0
        df.loc[df['Signal'] > 0, '_temp_position'] = 1   # Long
        df.loc[df['Signal'] < 0, '_temp_position'] = -1  # Short

        # Only set Target_Position when it CHANGES
        df['Target_Position'] = np.nan
        position_changes = df['_temp_position'] != df['_temp_position'].shift(1)
        df.loc[position_changes, 'Target_Position'] = df.loc[position_changes, '_temp_position']

        # Forward-fill to get position at close
        df['Position_At_Close'] = df['Target_Position'].ffill().fillna(0)

        # Drop temporary column
        df.drop(columns=['_temp_position'], inplace=True)

        # Default entry timing
        df['Entry_Time'] = 'close'
        df['Entry_Price'] = df['Close']

        # Return only the strategy period (after indicators are calculated)
        return df.loc[self.start_date:self.end_date]


if __name__ == "__main__":
    # Example usage
    import logging
    logging.basicConfig(level=logging.INFO)

    print("=" * 60)
    print("Testing Base Strategy Framework")
    print("=" * 60)

    # Example 1: Buy and Hold
    print("\n1. Buy and Hold Strategy")
    print("-" * 60)
    bh_strategy = BuyAndHoldStrategy()
    print(f"Parameters: {bh_strategy.get_parameter_summary()}")

    signals = bh_strategy('AAPL', '2024-01-01', '2024-12-31')
    if signals is not None:
        print(f"\nGenerated {len(signals)} signals")
        print("\nFirst 5 signals:")
        print(signals.head())
        print("\nLast 5 signals:")
        print(signals.tail())
        print(f"\nPosition distribution:")
        print(signals['Position'].value_counts())

    # Example 2: Moving Average Crossover
    print("\n2. Moving Average Crossover Strategy")
    print("-" * 60)
    ma_strategy = MovingAverageCrossoverStrategy(
        short_period=20,
        long_period=50
    )
    print(f"Parameters: {ma_strategy.get_parameter_summary()}")

    signals = ma_strategy('MSFT', '2024-01-01', '2024-12-31')
    if signals is not None:
        print(f"\nGenerated {len(signals)} signals")
        print("\nFirst 10 signals:")
        print(signals[['Close', 'MA_Short', 'MA_Long', 'Signal', 'Position']].head(10))
        print(f"\nSignal distribution:")
        print(signals['Signal'].value_counts())

        # Count transitions
        transitions = (signals['Signal'] != signals['Signal'].shift()).sum()
        print(f"\nNumber of signal transitions: {transitions}")
