# Testtube - Backtesting Engine

A backtesting engine for daily price series with intelligent caching and flexible strategy framework.

## Project Structure

```
quantstrat/
├── libs/
│   ├── cache.py         # Parquet-based caching with configurable staleness
│   ├── prices.py        # Price fetching with automatic caching
│   ├── volatility.py    # Volatility calculation functions
│   └── utils.py         # Utility functions
├── strategies/
│   └── base.py          # Abstract base class for strategies
├── scripts/             # Entry point scripts
│   └── example_backtest.py
└── README.md           # This file
```

## Features

### 1. Intelligent Caching (`libs/cache.py`)

- **Parquet storage**: Each stock stored in separate `.parquet` file for efficient I/O
- **Metadata tracking**: JSON metadata files track last update time, columns, date range
- **Stock-specific staleness**: Each stock can have different freshness status
- **Configurable via `CacheConfig`**:
  - `cache_dir`: Where to store cache files (default: `.cache/prices/`)
  - `staleness_days`: How old before refresh (default: 1 day, supports fractional like 0.5)
  - `history_days`: How many days to store (default: 252)
  - `columns`: Which OHLCV columns to cache (default: all)

### 2. Price Fetching (`libs/prices.py`)

- **Compatible with existing code**: `fetch_stock_prices()` matches optionspnl interface
- **Two main functions**:
  - `fetch_stock_prices(symbol, num_days)`: Get closing prices (for MA analysis)
  - `fetch_price_range(symbol, start_date, end_date)`: Get OHLCV data
- **Force refresh**: `force_refresh=True` parameter to bypass cache
- **Cache management**: `refresh_cache()`, `get_cache_info()`, `configure_cache()`

### 3. Volatility Functions (`libs/volatility.py`)

- **Multiple volatility estimators**:
  - `simple_vol()`: Standard deviation of returns (most common)
  - `parkinson_vol()`: High-Low range-based (more efficient)
  - `garman_klass_vol()`: OHLC-based (most accurate)
  - `ewma_vol()`: Exponentially-weighted volatility
  - `realized_vol()`: Sum of squared returns
- **Flexible configuration**: Use `partial()` or `create_vol_function()` to set parameters
- **Automatic annualization**: All functions return annualized volatility by default

### 4. Strategy Framework (`strategies/base.py`)

- **Abstract base class**: `BaseStrategy` with clear interface
- **Automatic volatility calculation**: Each strategy calculates rolling volatility
- **Simple usage**:
  ```python
  from functools import partial
  from libs.volatility import simple_vol

  strategy = MyStrategy('2024-01-01', '2024-12-31',
                       volatility_function=partial(simple_vol, N=30),
                       param1=value1)
  df = strategy('AAPL')  # Returns DataFrame with prices, volatility, signals, positions
  ```
- **Result stored in self.df**: DataFrame contains prices, volatility, and positions
- **Built-in examples**:
  - `BuyAndHoldStrategy`: Simple benchmark
  - `MovingAverageCrossoverStrategy`: MA crossover example

## Quick Start

### Basic Usage

```python
from libs.prices import fetch_stock_prices, configure_cache
from strategies.base import MovingAverageCrossoverStrategy

# Optional: Configure cache at startup
configure_cache(staleness_days=0.5, history_days=500)

# Fetch prices (uses cache automatically)
prices = fetch_stock_prices('AAPL', num_days=201)

# Run a strategy
strategy = MovingAverageCrossoverStrategy(
    '2024-01-01', '2024-12-31',
    short_period=20, long_period=50
)
signals = strategy('AAPL')
print(signals.head())
```

### Creating a Custom Strategy

```python
from strategies.base import BaseStrategy

class MyStrategy(BaseStrategy):
    def __init__(self, start_date, end_date, threshold=0.02, **kwargs):
        super().__init__(start_date, end_date, threshold=threshold, **kwargs)
        self.threshold = threshold

    def generate_signals(self, prices):
        df = prices.copy()

        # Your strategy logic here
        df['Returns'] = df['Close'].pct_change()
        df['Signal'] = 0
        df.loc[df['Returns'] > self.threshold, 'Signal'] = 1
        df.loc[df['Returns'] < -self.threshold, 'Signal'] = -1

        df['Position'] = df['Signal']

        # Return data for strategy period only
        return df.loc[self.start_date:self.end_date]

# Use it
strategy = MyStrategy('2024-01-01', '2024-12-31', threshold=0.03)
signals = strategy('AAPL')
```

## Answers to Design Questions

### Q1: Should staleness be stock-specific?

**A: YES - Implemented as stock-specific**

Each stock has its own metadata file (`{symbol}_meta.json`) that tracks when it was last updated. This means:
- Different stocks can have different freshness levels
- High-priority stocks can be refreshed more frequently
- You can manually refresh specific stocks without touching others

The global `staleness_days` config sets the threshold, but staleness is checked per-stock.

### Q2: Can we reuse/refactor caching code from optionspnl/libs?

**A: YES - Reused and enhanced**

The implementation is based on `optionspnl/libs/historical_cache.py` with improvements:

**Reused concepts**:
- Separate file per stock (not one monolithic file)
- Metadata tracking (last_updated, columns, etc.)
- `needs_update()` method for staleness checking
- Force refresh capability
- Configurable staleness threshold

**Enhancements**:
- **Parquet instead of JSON**: ~10x faster for time series data, smaller files
- **Cleaner separation**: Cache logic in `cache.py`, price fetching in `prices.py`
- **Better config**: `CacheConfig` class for easier configuration management
- **More flexible**: Fractional staleness days (e.g., 0.5 = 12 hours)

### Q3: How will cache configuration be implemented?

**A: Multiple approaches supported**

#### Approach 1: Direct configuration (Recommended for scripts)
```python
from libs.prices import configure_cache

# At the start of your script
configure_cache(
    cache_dir='/path/to/cache',
    staleness_days=0.5,
    history_days=500
)

# All subsequent calls use this config
prices = fetch_stock_prices('AAPL')
```

#### Approach 2: Via CacheConfig object (Recommended for libraries)
```python
from libs.cache import CacheConfig, PriceCache, set_default_cache

# Create config
config = CacheConfig(
    staleness_days=0.25,
    history_days=1000,
    columns=['Close', 'Volume']  # Only cache these
)

# Create and set cache
cache = PriceCache(config)
set_default_cache(cache)
```

#### Approach 3: JSON configuration (For production)
```python
import json
from libs.cache import CacheConfig, PriceCache, set_default_cache

# Load from config file
with open('config.json', 'r') as f:
    config_dict = json.load(f)

config = CacheConfig.from_dict(config_dict['cache'])
cache = PriceCache(config)
set_default_cache(cache)
```

Example `config.json`:
```json
{
  "cache": {
    "cache_dir": "/data/quantstrat/cache",
    "staleness_days": 1,
    "history_days": 252,
    "columns": ["Open", "High", "Low", "Close", "Volume"]
  },
  "backtest": {
    "start_date": "2023-01-01",
    "end_date": "2024-12-31"
  }
}
```

## Cache Management

### View cache status
```python
from libs.prices import get_cache_info

# All cached symbols
summary = get_cache_info()
print(summary)

# Specific symbol
info = get_cache_info('AAPL')
print(info)
```

### Force refresh
```python
from libs.prices import refresh_cache

# Refresh specific symbol
refresh_cache('AAPL')

# Clear entire cache
refresh_cache()
```

### Bypass cache for one call
```python
# Use cache by default, but force refresh this time
prices = fetch_stock_prices('AAPL', force_refresh=True)
```

## Testing

Each module has example usage in its `if __name__ == "__main__":` block:

```bash
# Test caching
python libs/cache.py

# Test price fetching
python libs/prices.py

# Test strategies
python strategies/base.py
```

## Next Steps (TBD)

1. **Backtester module** (`libs/backtester.py`):
   - Orchestrate strategy execution across multiple symbols
   - Track positions, calculate PnL
   - Handle transaction costs, slippage

2. **Performance analytics** (`libs/performance.py`):
   - Individual stock metrics (returns, Sharpe, max drawdown)
   - Portfolio-level aggregation
   - Comparison to benchmarks

3. **Entry point script** (`scripts/run_backtest.py`):
   - CLI interface
   - Load config from JSON
   - Generate reports

## Dependencies

- `pandas`: Data manipulation
- `yfinance`: Yahoo Finance data
- `pyarrow`: Parquet file support
- `numpy`: Numerical operations (from existing utils.py)

Install with:
```bash
pip install pandas yfinance pyarrow numpy
```

## Design Philosophy

1. **Separation of concerns**: Cache, fetching, and strategy are separate modules
2. **Reusable components**: Cache and price modules can be used independently
3. **Flexible configuration**: Multiple ways to configure based on use case
4. **Backward compatible**: `fetch_stock_prices()` works as drop-in replacement
5. **Performance**: Parquet files for fast I/O, intelligent caching to reduce API calls
