"""
Strategy Modifiers Module

This module provides the Strategy Modifier pattern - composable transformations
that modify strategy behavior without changing the backtester or core strategy logic.

Modifiers can change:
- Entry/exit timing (next open, next close, VWAP, etc.)
- Position sizing (scaling, capping)
- Signal filtering (day-of-week, minimum strength)
- Risk management (stop loss, take profit)
- Order types (limit orders)

All modifiers inherit from StrategyModifier and implement the transform() method.
"""

import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


class StrategyModifier(ABC):
    """
    Abstract base class for strategy modifiers.

    A modifier wraps a strategy and transforms its DataFrame in-place,
    modifying columns like Entry_Time, Entry_Price, Target_Position, etc.

    Modifiers can be chained using fluent API:
        strategy = DMA(50).enter_at_next_open().with_stop_loss(2.0)
    """

    def __init__(self, strategy):
        """
        Initialize modifier with a strategy.

        Args:
            strategy: BaseStrategy instance or another StrategyModifier
        """
        self.strategy = strategy

    @abstractmethod
    def transform(self, df):
        """
        Transform the strategy DataFrame in-place.

        Args:
            df (pd.DataFrame): Strategy DataFrame with columns:
                - Target_Position: Strategy's target position (set only on signal days)
                - Position_At_Close: Actual position at close (every day)
                - Entry_Time: 'close' (default), 'next_open', 'next_close', etc.
                - Entry_Price: Price at which entry occurs

        Returns:
            pd.DataFrame: Transformed DataFrame (modified in-place, but returned for chaining)
        """
        pass

    def __call__(self, symbol, start_date, end_date):
        """
        Run the wrapped strategy and apply the modifier transformation.

        Args:
            symbol (str): Stock ticker
            start_date (str): Start date
            end_date (str): End date

        Returns:
            pd.DataFrame: Transformed strategy DataFrame
        """
        # Run the underlying strategy (which may be another modifier or base strategy)
        df = self.strategy(symbol, start_date, end_date)

        if df is None or df.empty:
            return df

        # Apply the transformation (modifies df in-place)
        df = self.transform(df)

        # Store the modified df back to the wrapped strategy
        # This ensures that .df attribute has the fully transformed data
        self.strategy.df = df

        # Also store on self for direct access
        self.df = df

        return df

    def __getattr__(self, name):
        """
        Delegate attribute access to wrapped strategy.

        This allows modifiers to be transparent wrappers - any attributes
        not found on the modifier are looked up on the wrapped strategy.
        """
        return getattr(self.strategy, name)

    # =============================================================================
    # Modifier Chaining Methods (Fluent API)
    # =============================================================================
    # These methods enable chaining modifiers together, e.g.:
    #   DMA(50).enter_at_next_open().cap_position(1.0).with_stop_loss(2.0)

    def enter_at_next_open(self):
        """Wrap this modifier with EnterAtNextOpen."""
        return EnterAtNextOpen(self)

    def enter_at_next_close(self):
        """Wrap this modifier with EnterAtNextClose."""
        return EnterAtNextClose(self)

    def exit_at_next_open(self):
        """Wrap this modifier with ExitAtNextOpen."""
        return ExitAtNextOpen(self)

    def cap_position(self, max_position):
        """Wrap this modifier with CapPosition."""
        return CapPosition(self, max_position)

    def only_on_days(self, allowed_days):
        """Wrap this modifier with TradeOnlyOnDays."""
        return TradeOnlyOnDays(self, allowed_days)

    def minimum_signal_strength(self, min_strength):
        """Wrap this modifier with MinimumSignalStrength."""
        return MinimumSignalStrength(self, min_strength)

    def with_stop_loss(self, pct):
        """Wrap this modifier with WithStopLoss."""
        return WithStopLoss(self, pct)

    def with_take_profit(self, pct):
        """Wrap this modifier with WithTakeProfit."""
        return WithTakeProfit(self, pct)

    def limit_order(self, limit_pct):
        """Wrap this modifier with LimitOrder."""
        return LimitOrder(self, limit_pct)


# =============================================================================
# Timing Modifiers
# =============================================================================

class EnterAtNextOpen(StrategyModifier):
    """
    Modifier: Enter positions at next day's open price.

    When a signal is generated at close of day T, the entry occurs at
    open of day T+1 instead of close of day T.

    Example:
        strategy = DMA(200).enter_at_next_open()
    """

    def transform(self, df):
        """Transform to delay entry to next open."""
        # Find signal days where entry occurs
        is_signal_day = df['Target_Position'].notna()
        will_enter = is_signal_day & (df['Target_Position'] > 0)

        # Set entry metadata on signal day
        df.loc[will_enter, 'Entry_Time'] = 'next_open'

        # Entry price will be the next day's open
        for idx in df.index[will_enter]:
            idx_loc = df.index.get_loc(idx)
            if idx_loc + 1 < len(df):
                next_idx = df.index[idx_loc + 1]
                df.loc[idx, 'Entry_Price'] = df.loc[next_idx, 'Open']

        # CRITICAL: Delay position establishment by one day
        # This shifts Target_Position forward by 1 day, then forward fills
        new_position = df['Target_Position'].shift(1).fillna(method='ffill').fillna(0)
        df['Position_At_Close'] = new_position

        # For entry at next_open: exposure occurs when:
        # 1. We hold at close (Position_At_Close != 0), OR
        # 2. We will hold tomorrow (next day's Position_At_Close != 0), meaning we enter today at open
        # This captures both entry days (where we enter at open) and holding days and exit days
        current_position = df['Position_At_Close'] != 0
        next_position = df['Position_At_Close'].shift(-1).fillna(0) != 0
        df['Is_Exposure_Day'] = current_position | next_position

        # Mark for backtester
        df.attrs['has_next_open_entries'] = True

        return df


class EnterAtNextClose(StrategyModifier):
    """
    Modifier: Enter positions at next day's close price.

    When a signal is generated at close of day T, the entry occurs at
    close of day T+1 instead of close of day T (1-day lag).

    Example:
        strategy = DMA(200).enter_at_next_close()
    """

    def transform(self, df):
        """Transform to delay entry to next close."""
        # Find signal days where entry occurs
        is_signal_day = df['Target_Position'].notna()
        will_enter = is_signal_day & (df['Target_Position'] > 0)

        # Set entry metadata on signal day
        df.loc[will_enter, 'Entry_Time'] = 'next_close'

        # Entry price will be the next day's close
        for idx in df.index[will_enter]:
            idx_loc = df.index.get_loc(idx)
            if idx_loc + 1 < len(df):
                next_idx = df.index[idx_loc + 1]
                df.loc[idx, 'Entry_Price'] = df.loc[next_idx, 'Close']

        # CRITICAL: Delay position establishment by one day
        # This shifts Target_Position forward by 1 day, then forward fills
        new_position = df['Target_Position'].shift(1).fillna(method='ffill').fillna(0)
        df['Position_At_Close'] = new_position

        # For entry at next_close: exposure starts the day AFTER position is established
        # Signal on T -> Enter at close of T+1 -> Exposure starts T+2 (overnight holding)
        df['Is_Exposure_Day'] = df['Position_At_Close'].shift(1).fillna(0) != 0

        # Mark for backtester
        df.attrs['has_next_close_entries'] = True

        return df


class ExitAtNextOpen(StrategyModifier):
    """
    Modifier: Exit positions at next day's open price.

    When an exit signal is generated at close of day T, the exit occurs
    at open of day T+1 instead of close of day T.

    This extends the position holding period to the next open.

    Example:
        strategy = DMA(200).exit_at_next_open()
    """

    def transform(self, df):
        """Transform to set exit timing to next open."""
        # Mark days where we exit (Position_At_Close goes from non-zero to zero)
        position_at_prior_close = df['Position_At_Close'].shift(1).fillna(0)
        is_exit = (position_at_prior_close != 0) & (df['Position_At_Close'] == 0)

        # Set exit flag
        df['Exit_Time'] = 'close'  # default
        df.loc[is_exit, 'Exit_Time'] = 'next_open'

        # Store this in df.attrs for backtester
        df.attrs['has_exit_at_open'] = True

        return df


# =============================================================================
# Risk Management Modifiers
# =============================================================================

class CapPosition(StrategyModifier):
    """
    Modifier: Cap position sizes at a maximum value.

    Ensures |Target_Position| <= max_position.

    Example:
        strategy = DMA(200).cap_position(1.0)  # Max position = 1
    """

    def __init__(self, strategy, max_position):
        """
        Initialize CapPosition modifier.

        Args:
            strategy: Wrapped strategy
            max_position (float): Maximum absolute position size
        """
        super().__init__(strategy)
        self.max_position = abs(max_position)

    def transform(self, df):
        """Cap all target positions."""
        # Cap Target_Position where it's not NaN
        mask = df['Target_Position'].notna()
        df.loc[mask, 'Target_Position'] = df.loc[mask, 'Target_Position'].clip(
            -self.max_position, self.max_position
        )

        # Also cap Position_At_Close
        df['Position_At_Close'] = df['Position_At_Close'].clip(
            -self.max_position, self.max_position
        )

        return df


# =============================================================================
# Signal Filtering Modifiers
# =============================================================================

class TradeOnlyOnDays(StrategyModifier):
    """
    Modifier: Only allow trades on specific days of the week.

    Filters out signals that don't occur on allowed days.

    Example:
        strategy = DMA(200).only_on_days(['Monday', 'Wednesday', 'Friday'])
    """

    def __init__(self, strategy, allowed_days):
        """
        Initialize TradeOnlyOnDays modifier.

        Args:
            strategy: Wrapped strategy
            allowed_days (list): List of day names (e.g., ['Monday', 'Friday'])
        """
        super().__init__(strategy)
        self.allowed_days = [day.lower() for day in allowed_days]

    def transform(self, df):
        """Filter signals by day of week."""
        # Get day names
        df['_day_name'] = df.index.day_name()

        # Filter Target_Position to only allowed days
        signal_days = df['Target_Position'].notna()
        disallowed_days = ~df['_day_name'].str.lower().isin(self.allowed_days)

        # Set Target_Position to NaN for disallowed days
        df.loc[signal_days & disallowed_days, 'Target_Position'] = np.nan

        # Recalculate Position_At_Close based on filtered signals
        df['Position_At_Close'] = df['Target_Position'].fillna(method='ffill').fillna(0)

        # Clean up temporary column
        df.drop(columns=['_day_name'], inplace=True)

        return df


class MinimumSignalStrength(StrategyModifier):
    """
    Modifier: Only trade when signal strength exceeds a threshold.

    For strategies that output real-valued signals, this filters out
    weak signals below a minimum absolute value.

    Example:
        strategy = MyStrategy().minimum_signal_strength(0.1)
    """

    def __init__(self, strategy, min_strength):
        """
        Initialize MinimumSignalStrength modifier.

        Args:
            strategy: Wrapped strategy
            min_strength (float): Minimum absolute signal strength
        """
        super().__init__(strategy)
        self.min_strength = abs(min_strength)

    def transform(self, df):
        """Filter signals by minimum strength."""
        if 'Signal' not in df.columns:
            logger.warning("MinimumSignalStrength: No 'Signal' column found, skipping")
            return df

        # Find signal days
        signal_days = df['Target_Position'].notna()

        # Filter by minimum signal strength
        weak_signals = signal_days & (df['Signal'].abs() < self.min_strength)

        # Set Target_Position to 0 for weak signals
        df.loc[weak_signals, 'Target_Position'] = 0

        # Recalculate Position_At_Close
        df['Position_At_Close'] = df['Target_Position'].fillna(method='ffill').fillna(0)

        return df


# =============================================================================
# Risk Management Modifiers
# =============================================================================

class WithStopLoss(StrategyModifier):
    """
    Modifier: Exit position if loss exceeds a percentage threshold.

    Monitors cumulative PnL and exits when loss from entry exceeds stop_pct.

    Example:
        strategy = DMA(200).with_stop_loss(pct=2.0)  # 2% stop loss
    """

    def __init__(self, strategy, pct):
        """
        Initialize WithStopLoss modifier.

        Args:
            strategy: Wrapped strategy
            pct (float): Stop loss percentage (e.g., 2.0 = 2%)
        """
        super().__init__(strategy)
        self.stop_pct = abs(pct)

    def transform(self, df):
        """Apply stop loss logic."""
        # Track entry price and implement stop loss
        df['Entry_Price_Ref'] = np.nan
        df['Stop_Loss_Hit'] = False

        entry_price = None
        position = 0

        for idx in df.index:
            current_position = df.loc[idx, 'Position_At_Close']

            # Detect entry
            if position == 0 and current_position != 0:
                entry_price = df.loc[idx, 'Entry_Price']
                df.loc[idx, 'Entry_Price_Ref'] = entry_price

            # Check stop loss if in position
            if current_position != 0 and entry_price is not None:
                current_price = df.loc[idx, 'Close']

                # Calculate loss percentage
                if current_position > 0:  # Long position
                    loss_pct = ((current_price - entry_price) / entry_price) * 100
                else:  # Short position
                    loss_pct = ((entry_price - current_price) / entry_price) * 100

                # Exit if stop loss hit
                if loss_pct < -self.stop_pct:
                    df.loc[idx, 'Position_At_Close'] = 0
                    df.loc[idx, 'Stop_Loss_Hit'] = True
                    entry_price = None

            # Update for next iteration
            if df.loc[idx, 'Position_At_Close'] == 0:
                entry_price = None

            position = df.loc[idx, 'Position_At_Close']

        return df


class WithTakeProfit(StrategyModifier):
    """
    Modifier: Exit position if profit exceeds a percentage threshold.

    Monitors cumulative PnL and exits when profit from entry exceeds target_pct.

    Example:
        strategy = DMA(200).with_take_profit(pct=5.0)  # 5% take profit
    """

    def __init__(self, strategy, pct):
        """
        Initialize WithTakeProfit modifier.

        Args:
            strategy: Wrapped strategy
            pct (float): Take profit percentage (e.g., 5.0 = 5%)
        """
        super().__init__(strategy)
        self.target_pct = abs(pct)

    def transform(self, df):
        """Apply take profit logic."""
        # Track entry price and implement take profit
        df['Take_Profit_Hit'] = False

        entry_price = None
        position = 0

        for idx in df.index:
            current_position = df.loc[idx, 'Position_At_Close']

            # Detect entry
            if position == 0 and current_position != 0:
                entry_price = df.loc[idx, 'Entry_Price']

            # Check take profit if in position
            if current_position != 0 and entry_price is not None:
                current_price = df.loc[idx, 'Close']

                # Calculate profit percentage
                if current_position > 0:  # Long position
                    profit_pct = ((current_price - entry_price) / entry_price) * 100
                else:  # Short position
                    profit_pct = ((entry_price - current_price) / entry_price) * 100

                # Exit if take profit hit
                if profit_pct > self.target_pct:
                    df.loc[idx, 'Position_At_Close'] = 0
                    df.loc[idx, 'Take_Profit_Hit'] = True
                    entry_price = None

            # Update for next iteration
            if df.loc[idx, 'Position_At_Close'] == 0:
                entry_price = None

            position = df.loc[idx, 'Position_At_Close']

        return df


# =============================================================================
# Order Type Modifiers
# =============================================================================

class LimitOrder(StrategyModifier):
    """
    Modifier: Enter using limit orders instead of market orders.

    Sets a limit price at a percentage below/above current price.
    Entry only occurs if price reaches the limit.

    Example:
        strategy = DMA(200).limit_order(limit_pct=0.5)  # 0.5% below for long
    """

    def __init__(self, strategy, limit_pct):
        """
        Initialize LimitOrder modifier.

        Args:
            strategy: Wrapped strategy
            limit_pct (float): Limit percentage (e.g., 0.5 = 0.5% below current for long)
        """
        super().__init__(strategy)
        self.limit_pct = abs(limit_pct)

    def transform(self, df):
        """Apply limit order logic."""
        # Find entry signals
        signal_days = df['Target_Position'].notna()
        will_enter = signal_days & (df['Target_Position'] != 0)

        # Set limit prices
        # For long: limit = close * (1 - limit_pct/100)
        # For short: limit = close * (1 + limit_pct/100)
        df['Limit_Price'] = np.nan

        long_entries = will_enter & (df['Target_Position'] > 0)
        short_entries = will_enter & (df['Target_Position'] < 0)

        df.loc[long_entries, 'Limit_Price'] = df.loc[long_entries, 'Close'] * (1 - self.limit_pct / 100)
        df.loc[short_entries, 'Limit_Price'] = df.loc[short_entries, 'Close'] * (1 + self.limit_pct / 100)

        df.loc[will_enter, 'Entry_Type'] = 'limit'

        # Mark that this strategy uses limit orders
        df.attrs['has_limit_orders'] = True

        return df
