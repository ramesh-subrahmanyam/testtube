"""
Technical Indicators Module

This module provides technical indicator functions for use in trading strategies.
All indicators use a factory pattern to allow for easy parameterization.
The returned functions now take symbol and price series as arguments.

Example:
    >>> sma_200 = SMA(200)
    >>> ma = sma_200('AAPL', prices['Close'])
"""

import pandas as pd
import logging

logger = logging.getLogger(__name__)


def SMA(N):
    """
    Simple Moving Average (SMA) indicator factory.

    This function returns a callable that computes the N-day simple moving average.
    Uses a decorator pattern to allow easy parameterization.

    Args:
        N (int): The number of periods for the moving average

    Returns:
        callable: A function that takes symbol and pandas Series and returns the N-day SMA

    Example:
        >>> # Create a 200-day SMA calculator
        >>> sma_200 = SMA(200)
        >>>
        >>> # Apply it to price data
        >>> df['MA_200'] = sma_200('AAPL', df['Close'])
        >>>
        >>> # Or use inline
        >>> df['MA_50'] = SMA(50)('AAPL', df['Close'])
    """
    def compute_sma(symbol, series):
        """
        Compute the N-day simple moving average.

        Args:
            symbol (str): Stock ticker symbol (e.g., 'AAPL')
            series (pd.Series): Time series data (typically price data)

        Returns:
            pd.Series: N-day simple moving average
        """
        if not isinstance(series, pd.Series):
            raise TypeError(f"Expected pandas Series, got {type(series)}")

        result = series.rolling(window=N).mean()
        logger.debug(f"Computed {N}-day SMA for {symbol} with {len(result)} data points")

        return result

    # Add metadata to the function for introspection
    compute_sma.__name__ = f'SMA_{N}'
    compute_sma.__doc__ = f"""Compute {N}-day Simple Moving Average.

    Args:
        symbol (str): Stock ticker symbol
        series (pd.Series): Time series data

    Returns:
        pd.Series: {N}-day simple moving average
    """
    compute_sma.period = N
    compute_sma.indicator_type = 'SMA'

    return compute_sma


def EMA(N):
    """
    Exponential Moving Average (EMA) indicator factory.

    This function returns a callable that computes the N-day exponential moving average.
    Uses a decorator pattern to allow easy parameterization.

    Args:
        N (int): The number of periods for the moving average

    Returns:
        callable: A function that takes symbol and pandas Series and returns the N-day EMA

    Example:
        >>> # Create a 20-day EMA calculator
        >>> ema_20 = EMA(20)
        >>>
        >>> # Apply it to price data
        >>> df['EMA_20'] = ema_20('AAPL', df['Close'])
    """
    def compute_ema(symbol, series):
        """
        Compute the N-day exponential moving average.

        Args:
            symbol (str): Stock ticker symbol (e.g., 'AAPL')
            series (pd.Series): Time series data (typically price data)

        Returns:
            pd.Series: N-day exponential moving average
        """
        if not isinstance(series, pd.Series):
            raise TypeError(f"Expected pandas Series, got {type(series)}")

        result = series.ewm(span=N, adjust=False).mean()
        logger.debug(f"Computed {N}-day EMA for {symbol} with {len(result)} data points")

        return result

    # Add metadata to the function for introspection
    compute_ema.__name__ = f'EMA_{N}'
    compute_ema.__doc__ = f"""Compute {N}-day Exponential Moving Average.

    Args:
        symbol (str): Stock ticker symbol
        series (pd.Series): Time series data

    Returns:
        pd.Series: {N}-day exponential moving average
    """
    compute_ema.period = N
    compute_ema.indicator_type = 'EMA'

    return compute_ema


def RSI(N):
    """
    Relative Strength Index (RSI) indicator factory.

    This function returns a callable that computes the N-period RSI.
    Uses a decorator pattern to allow easy parameterization.

    Args:
        N (int): The number of periods for the RSI calculation

    Returns:
        callable: A function that takes symbol and pandas Series and returns the N-period RSI

    Example:
        >>> # Create a 14-period RSI calculator
        >>> rsi_14 = RSI(14)
        >>>
        >>> # Apply it to price data
        >>> df['RSI'] = rsi_14('AAPL', df['Close'])
    """
    def compute_rsi(symbol, series):
        """
        Compute the N-period Relative Strength Index.

        Args:
            symbol (str): Stock ticker symbol (e.g., 'AAPL')
            series (pd.Series): Time series data (typically price data)

        Returns:
            pd.Series: N-period RSI (values between 0 and 100)
        """
        if not isinstance(series, pd.Series):
            raise TypeError(f"Expected pandas Series, got {type(series)}")

        # Calculate price changes
        delta = series.diff()

        # Separate gains and losses
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        # Calculate average gain and loss using exponential moving average
        avg_gain = gain.ewm(span=N, adjust=False).mean()
        avg_loss = loss.ewm(span=N, adjust=False).mean()

        # Calculate RS and RSI
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        logger.debug(f"Computed {N}-period RSI for {symbol} with {len(rsi)} data points")

        return rsi

    # Add metadata to the function for introspection
    compute_rsi.__name__ = f'RSI_{N}'
    compute_rsi.__doc__ = f"""Compute {N}-period Relative Strength Index.

    Args:
        symbol (str): Stock ticker symbol
        series (pd.Series): Time series data

    Returns:
        pd.Series: {N}-period RSI (values between 0 and 100)
    """
    compute_rsi.period = N
    compute_rsi.indicator_type = 'RSI'

    return compute_rsi


if __name__ == "__main__":
    # Example usage and testing
    import numpy as np

    print("=" * 60)
    print("Testing Technical Indicators")
    print("=" * 60)

    # Create sample price data
    dates = pd.date_range('2024-01-01', periods=300, freq='D')
    prices = pd.Series(
        100 + np.cumsum(np.random.randn(300) * 2),
        index=dates,
        name='Close'
    )

    print("\nSample price data (first 10 days):")
    print(prices.head(10))

    # Test SMA with different periods
    print("\n" + "=" * 60)
    print("Testing SMA (Simple Moving Average)")
    print("=" * 60)

    # Create SMA calculators
    sma_20 = SMA(20)
    sma_50 = SMA(50)
    sma_200 = SMA(200)

    print(f"\nCreated SMA calculators:")
    print(f"  SMA(20): {sma_20.__name__} - {sma_20.indicator_type}, period={sma_20.period}")
    print(f"  SMA(50): {sma_50.__name__} - {sma_50.indicator_type}, period={sma_50.period}")
    print(f"  SMA(200): {sma_200.__name__} - {sma_200.indicator_type}, period={sma_200.period}")

    # Compute moving averages
    ma_20 = sma_20(prices)
    ma_50 = sma_50(prices)
    ma_200 = sma_200(prices)

    print(f"\n20-day SMA (last 10 days):")
    print(ma_20.tail(10))

    print(f"\n50-day SMA (last 10 days):")
    print(ma_50.tail(10))

    print(f"\n200-day SMA (last 10 days):")
    print(ma_200.tail(10))

    # Test EMA
    print("\n" + "=" * 60)
    print("Testing EMA (Exponential Moving Average)")
    print("=" * 60)

    ema_20 = EMA(20)
    ema_result = ema_20(prices)

    print(f"\nCreated EMA calculator:")
    print(f"  EMA(20): {ema_20.__name__} - {ema_20.indicator_type}, period={ema_20.period}")

    print(f"\n20-day EMA (last 10 days):")
    print(ema_result.tail(10))

    # Compare SMA vs EMA
    print("\n" + "=" * 60)
    print("Comparison: SMA(20) vs EMA(20)")
    print("=" * 60)

    comparison = pd.DataFrame({
        'Price': prices,
        'SMA_20': ma_20,
        'EMA_20': ema_result,
        'Diff': ma_20 - ema_result
    })

    print("\nLast 10 days:")
    print(comparison.tail(10))

    print("\nTest completed successfully!")
