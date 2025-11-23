# Quantstrat Backtesting Engine - Design Document

## Overview

This backtesting engine is designed for testing trading strategies on daily price series. It fetches stock prices, runs strategies to determine daily positions, and presents performance results at both individual stock and portfolio levels.

## Architecture

The system is organized into the following modules:

### 1. Data Layer
- **libs/prices.py**: Handles fetching and retrieving stock price data
- **libs/cache.py**: Manages caching of price data in parquet files

### 2. Strategy Layer
- **strategies/base.py**: Abstract base class defining the strategy interface
- **strategies/dma.py**: Concrete implementation of a Daily Moving Average strategy
- **signals/technical.py**: Technical indicators (SMA, etc.)

### 3. Calculation Layer
- **libs/volatility.py**: Volatility calculation functions
- **libs/backtester.py**: Core backtesting engine with position sizing and slippage

### 4. Analysis Layer
- **libs/performance.py**: Performance statistics calculation

### 5. Execution Layer
- **scripts/**: Main execution scripts

## Component Details

### libs/cache.py
**Responsibilities:**
- Store each stock's data in a separate parquet file
- Track cache staleness (configurable days to go stale)
- Provide `refresh_cache()` method to force refresh
- Handle cache configuration parameters

**Key Methods:**
- `get(symbol, start_date, end_date)`: Retrieve cached data if fresh
- `set(symbol, data)`: Store data in cache
- `is_stale(symbol)`: Check if cached data needs refresh
- `refresh_cache(symbol)`: Force refresh of cache

### libs/prices.py
**Responsibilities:**
- Fetch stock prices from Yahoo Finance
- Interface with cache layer
- Reuse `fetch_stock_prices` from `../optionspnl/libs/dma_analysis.py`

**Key Methods:**
- `get_prices(symbol, start_date, end_date)`: Main interface to get prices
  - First checks cache (via libs/cache.py)
  - If stale or missing, fetches from Yahoo Finance
  - Updates cache with fresh data
  - Returns price DataFrame

### libs/volatility.py
**Responsibilities:**
- Calculate rolling volatility metrics

**Key Functions:**
- `simple_vol(window)`: Returns a function that computes trailing N-day standard deviation of daily returns
  - Daily return = (today's close / yesterday's close) - 1
  - Uses partial application pattern: `simple_vol(20)` returns a 20-day volatility calculator

### signals/technical.py
**Responsibilities:**
- Implement technical indicators as signals

**Key Functions:**
- `SMA(window)`: Returns a function that computes N-day simple moving average
  - Uses decorator pattern for parameterization
  - Returns NaN for first N days (insufficient data)
  - Example: `SMA(200)` creates a 200-day moving average calculator

### strategies/base.py
**Responsibilities:**
- Define abstract interface for all strategies
- Handle common strategy operations

**Class: Strategy (Abstract Base Class)**

**Attributes:**
- `self.df`: DataFrame containing prices, volatility, and positions

**Methods:**
- `__init__(volatility_function, **strategy_params)`:
  - Store volatility calculation function
  - Store strategy-specific parameters

- `__call__(symbol, start_date, end_date)`:
  - Fetch prices using `libs/prices.get_prices()`
  - Calculate rolling volatility using `volatility_function`
  - Generate positions series (implemented by subclasses)
  - Trim initial NaN period from signal warm-up
  - Validate no NaNs exist in middle of series
  - If middle NaNs found: create error file with details
  - Store prices, volatility, positions in `self.df`

### strategies/dma.py
**Responsibilities:**
- Implement Daily Moving Average strategy

**Class: DMAStrategy (extends Strategy)**

**Position Logic:**
- If price >= SMA: position = 1 (long)
- If price < SMA: position = 0 (flat)

### libs/backtester.py
**Responsibilities:**
- Execute backtest with position sizing and slippage
- Calculate P&L

**Class: Backtester**

**Attributes:**
- `strategy`: Strategy object
- `dollar_size`: Dollar amount for position sizing
- `slipped_performance`: Performance dict with slippage
- `unslipped_performance`: Performance dict without slippage

**Methods:**
- `__init__(strategy, dollar_size)`: Initialize with strategy and sizing

- `__call__(symbol, start_date, end_date, slippage_bps)`:
  1. Run strategy to get positions and prices
  2. Calculate dollar volatility = return volatility × price
  3. Calculate vol-adjusted positions = raw positions × (dollar_size / dollar_volatility)
  4. Calculate unslipped P&L (vectorized)
  5. Calculate slippage per day = abs(position change) × price × slippage_bps / 10000
  6. Calculate slipped P&L = unslipped P&L - slippage
  7. Call `stats()` on both P&L series
  8. Store results in `slipped_performance` and `unslipped_performance`

### libs/performance.py
**Responsibilities:**
- Calculate performance statistics

**Functions:**
- `stats(pnl_series)`: Returns dictionary with:
  - `sharpe`: Sharpe ratio
  - `total_pnl`: Cumulative P&L
  - `num_trades`: Number of trades executed
  - `mean_pnl_per_trade`: Average P&L per trade

## Main Workflow

### Typical Usage Flow

```python
# 1. Import modules
from libs.volatility import simple_vol
from signals.technical import SMA
from strategies.dma import DMAStrategy
from libs.backtester import Backtester

# 2. Configure strategy components
vol_func = simple_vol(20)  # 20-day volatility
signal_func = SMA(200)      # 200-day moving average

# 3. Create strategy instance
strategy = DMAStrategy(
    volatility_function=vol_func,
    signal_function=signal_func
)

# 4. Create backtester
backtester = Backtester(
    strategy=strategy,
    dollar_size=100000  # $100k position sizing
)

# 5. Run backtest
backtester(
    symbol='AAPL',
    start_date='2010-01-01',
    end_date='2024-12-31',
    slippage_bps=5  # 5 basis points slippage
)

# 6. Access results
print("Slipped Performance:", backtester.slipped_performance)
print("Unslipped Performance:", backtester.unslipped_performance)
```

## Detailed Method Call Sequence

### During Strategy Execution (strategy.__call__)

1. **Data Retrieval:**
   ```
   strategy.__call__(symbol, start_date, end_date)
     └─> libs/prices.get_prices(symbol, start_date, end_date)
           ├─> libs/cache.get(symbol, start_date, end_date)
           ├─> If stale/missing: fetch_stock_prices(symbol, start_date, end_date)
           └─> libs/cache.set(symbol, data)
   ```

2. **Signal & Position Generation:**
   ```
   - Apply volatility_function to price data → volatility series
   - Apply signal function (e.g., SMA) to price data → signal series
   - Apply position logic to signal → raw positions series
   - Trim initial NaN period from positions
   - Validate no middle NaNs (fail with error file if found)
   - Store in self.df: ['price', 'volatility', 'position']
   ```

### During Backtest Execution (backtester.__call__)

1. **Strategy Execution:**
   ```
   backtester.__call__(symbol, start_date, end_date, slippage_bps)
     └─> strategy(symbol, start_date, end_date)
           → Populates strategy.df with prices, volatility, positions
   ```

2. **Position Sizing:**
   ```
   - Extract prices, volatility, positions from strategy.df
   - Calculate dollar_volatility = volatility × price
   - Calculate sized_positions = positions × (dollar_size / dollar_volatility)
   ```

3. **P&L Calculation (Vectorized):**
   ```
   - Calculate returns = price.pct_change()
   - Calculate unslipped_pnl = sized_positions.shift(1) × returns × price
   - Calculate position_changes = sized_positions.diff().abs()
   - Calculate slippage = position_changes × price × (slippage_bps / 10000)
   - Calculate slipped_pnl = unslipped_pnl - slippage
   ```

4. **Performance Analysis:**
   ```
   backtester.unslipped_performance = libs/performance.stats(unslipped_pnl)
   backtester.slipped_performance = libs/performance.stats(slipped_pnl)
   ```

## Data Flow Diagram

```
┌─────────────────┐
│  Yahoo Finance  │
└────────┬────────┘
         │
         ↓
    ┌────────┐     ┌────────────┐
    │ Cache  │←────│ prices.py  │
    └────────┘     └──────┬─────┘
                          │
                          ↓
                   ┌──────────────┐
                   │  Strategy    │
                   │              │
    ┌──────────┐   │ - volatility │
    │ signals/ │──→│ - signal     │
    │technical │   │ - positions  │
    └──────────┘   └──────┬───────┘
                          │
                          ↓
                   ┌──────────────┐
                   │  Backtester  │
                   │              │
                   │ - sizing     │
                   │ - P&L calc   │
                   │ - slippage   │
                   └──────┬───────┘
                          │
                          ↓
                   ┌──────────────┐
                   │ Performance  │
                   │   Stats      │
                   └──────────────┘
```

## Key Design Patterns

1. **Partial Application**: Volatility and signal functions use partial application to create parameterized functions
   - `simple_vol(20)` returns a 20-day volatility calculator
   - `SMA(200)` returns a 200-day moving average calculator

2. **Strategy Pattern**: Abstract base class allows multiple strategy implementations
   - Base class handles common operations (data fetching, validation)
   - Subclasses implement specific position logic

3. **Callable Classes**: Strategies and Backtester use `__call__` for intuitive usage
   - `strategy(symbol, start_date, end_date)`
   - `backtester(symbol, start_date, end_date, slippage_bps)`

4. **Caching Layer**: Separates data fetching from data storage
   - Improves performance
   - Reduces API calls
   - Configurable staleness

5. **Vectorization**: All calculations use pandas vectorized operations for efficiency
   - No loops for P&L calculation
   - No loops for position sizing
   - No loops for slippage calculation

## Configuration Strategy

Cache configuration is handled through parameters passed to the cache module:
- `days_to_stale`: Number of days before cache is considered stale
- Cache location: Configurable directory for parquet files
- Per-stock configuration: Can be extended to make staleness stock-specific

Top-level scripts pass configuration down to library components through initialization parameters rather than global configuration files.

## Error Handling

1. **Middle NaN Detection**: If NaNs appear in middle of signal series:
   - Create error file in current directory
   - Include signal function name
   - Report number of NaN values
   - List specific dates with NaN values

2. **Cache Misses**: Automatically handled by fetching from Yahoo Finance

3. **Data Validation**: Strategy validates data completeness before proceeding

## Extension Points

1. **New Strategies**: Subclass `Strategy` and implement position logic
2. **New Signals**: Add functions to `signals/technical.py`
3. **New Volatility Measures**: Add functions to `libs/volatility.py`
4. **Custom Performance Metrics**: Extend `stats()` function
5. **Portfolio-Level Analysis**: Can aggregate results from multiple symbol backtests
