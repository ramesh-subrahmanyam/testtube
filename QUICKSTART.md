# Quantstrat Quick Start Guide

## Installation

```bash
cd /Users/ramesh/code/quantstrat
pip install pandas yfinance pyarrow numpy
```

## 30-Second Test

```bash
# Test that everything works
python scripts/example_backtest.py
```

## Basic Usage (5 minutes)

### 1. Fetch Prices with Caching

```python
from libs import fetch_stock_prices, configure_cache

# Configure cache (do this once at startup)
configure_cache(staleness_days=1, history_days=252)

# Fetch prices - automatically cached
prices = fetch_stock_prices('AAPL', num_days=201)
print(prices.tail())
```

### 2. Run a Simple Strategy

```python
from strategies import BuyAndHoldStrategy

# Create strategy
strategy = BuyAndHoldStrategy('2024-01-01', '2024-12-31')

# Run on a symbol
df = strategy('AAPL')

# View results
print(df[['Close', 'Volatility', 'Position']].tail())
```

### 3. Use Moving Average Strategy

```python
from strategies import MovingAverageCrossoverStrategy

# Create MA crossover strategy (20/50 day)
strategy = MovingAverageCrossoverStrategy(
    '2024-01-01', '2024-12-31',
    short_period=20,
    long_period=50
)

# Run on multiple stocks
for symbol in ['AAPL', 'MSFT', 'GOOGL']:
    df = strategy(symbol)
    if df is not None:
        signals = (df['Signal'] != df['Signal'].shift()).sum()
        print(f"{symbol}: {signals} signal changes")
```

### 4. Create Your Own Strategy

```python
from strategies import BaseStrategy

class MyStrategy(BaseStrategy):
    def __init__(self, start_date, end_date, threshold=0.02):
        super().__init__(start_date, end_date, threshold=threshold)
        self.threshold = threshold

    def generate_signals(self, prices):
        df = prices.copy()

        # Your logic here
        df['Returns'] = df['Close'].pct_change()
        df['Signal'] = 0
        df.loc[df['Returns'] > self.threshold, 'Signal'] = 1
        df.loc[df['Returns'] < -self.threshold, 'Signal'] = -1
        df['Position'] = df['Signal']

        return df.loc[self.start_date:self.end_date]

# Use it
strategy = MyStrategy('2024-01-01', '2024-12-31', threshold=0.03)
df = strategy('AAPL')
```

### 5. Use Different Volatility Functions

```python
from strategies import MovingAverageCrossoverStrategy
from libs.volatility import garman_klass_vol
from functools import partial

# Use Garman-Klass volatility (more accurate than simple vol)
strategy = MovingAverageCrossoverStrategy(
    '2024-01-01', '2024-12-31',
    volatility_function=partial(garman_klass_vol, N=30),
    short_period=20,
    long_period=50
)

df = strategy('AAPL')
print(df[['Close', 'Volatility']].tail())
```

## Available Volatility Functions

```python
from libs.volatility import (
    simple_vol,         # Standard deviation (most common)
    parkinson_vol,      # High-Low range based
    garman_klass_vol,   # OHLCV based (most accurate)
    ewma_vol,           # Exponentially weighted
    realized_vol        # Sum of squared returns
)
from functools import partial

# Create volatility calculators
vol_20 = partial(simple_vol, N=20)        # 20-day simple
vol_30 = partial(garman_klass_vol, N=30)  # 30-day GK
vol_ewma = partial(ewma_vol, span=20)     # EWMA with span=20

# Use in strategy
strategy = MyStrategy('2024-01-01', '2024-12-31',
                     volatility_function=vol_30)
```

## Cache Management

```python
from libs import get_cache_info, refresh_cache

# View all cached stocks
print(get_cache_info())

# View specific stock
print(get_cache_info('AAPL'))

# Force refresh one stock
refresh_cache('AAPL')

# Clear entire cache
refresh_cache()
```

## Common Patterns

### Pattern 1: Backtest Multiple Strategies

```python
from strategies import BuyAndHoldStrategy, MovingAverageCrossoverStrategy

strategies = {
    'BuyHold': BuyAndHoldStrategy('2024-01-01', '2024-12-31'),
    'MA_20_50': MovingAverageCrossoverStrategy('2024-01-01', '2024-12-31',
                                               short_period=20, long_period=50),
    'MA_10_30': MovingAverageCrossoverStrategy('2024-01-01', '2024-12-31',
                                               short_period=10, long_period=30),
}

symbol = 'AAPL'
results = {}

for name, strategy in strategies.items():
    df = strategy(symbol)
    if df is not None:
        results[name] = df
        print(f"{name}: Final position = {df['Position'].iloc[-1]}")
```

### Pattern 2: Scan for Signals

```python
from strategies import MovingAverageCrossoverStrategy

strategy = MovingAverageCrossoverStrategy('2024-01-01', '2024-12-31',
                                          short_period=20, long_period=50)

symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META']

for symbol in symbols:
    df = strategy(symbol)
    if df is not None:
        current_signal = df['Signal'].iloc[-1]
        if current_signal == 1:
            print(f"{symbol}: LONG signal")
        elif current_signal == -1:
            print(f"{symbol}: SHORT signal")
```

### Pattern 3: Configuration from JSON

```python
import json
from libs.cache import CacheConfig, PriceCache, set_default_cache
from strategies import MovingAverageCrossoverStrategy

# config.json
config_json = '''
{
    "cache": {
        "staleness_days": 0.5,
        "history_days": 500,
        "columns": ["Open", "High", "Low", "Close", "Volume"]
    },
    "strategy": {
        "type": "ma_crossover",
        "short_period": 20,
        "long_period": 50
    }
}
'''

# Load config
config = json.loads(config_json)

# Setup cache
cache_config = CacheConfig.from_dict(config['cache'])
cache = PriceCache(cache_config)
set_default_cache(cache)

# Create strategy
strategy = MovingAverageCrossoverStrategy(
    '2024-01-01', '2024-12-31',
    **config['strategy']
)
```

## Accessing Strategy Results

After running a strategy, `self.df` contains:

```python
strategy = MovingAverageCrossoverStrategy('2024-01-01', '2024-12-31')
df = strategy('AAPL')

# df contains:
# - Open, High, Low, Close, Volume  (price data)
# - Volatility                      (calculated volatility)
# - MA_Short, MA_Long              (strategy indicators)
# - Signal                          (1=long, 0=flat, -1=short)
# - Position                        (actual position)

# Access the data
print(df.columns)
print(df.tail())

# Filter by signal
long_days = df[df['Signal'] == 1]
print(f"Long on {len(long_days)} days")

# Calculate returns (for next step: performance analytics)
df['Returns'] = df['Close'].pct_change()
df['Strategy_Returns'] = df['Returns'] * df['Position'].shift(1)
```

## Testing Individual Modules

```bash
# Test cache
python libs/cache.py

# Test prices
python libs/prices.py

# Test volatility
python libs/volatility.py

# Test strategies
python strategies/base.py

# Run complete example
python scripts/example_backtest.py
```

## File Locations

```
/Users/ramesh/code/quantstrat/
├── libs/
│   ├── cache.py          # Caching engine
│   ├── prices.py         # Price fetching
│   ├── volatility.py     # Volatility functions
│   └── utils.py          # Utilities
├── strategies/
│   └── base.py           # Strategy base class + examples
├── scripts/
│   └── example_backtest.py
└── README.md             # Full documentation
```

## Getting Help

- **Full docs**: [README.md](README.md)
- **Implementation details**: [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)
- **Complete checklist**: [COMPLETE_IMPLEMENTATION.md](COMPLETE_IMPLEMENTATION.md)
- **Run examples**: `python scripts/example_backtest.py`

## Next Steps After Testing

1. Implement performance metrics (Sharpe ratio, max drawdown)
2. Create backtester.py to orchestrate multi-stock runs
3. Add position_manager.py for PnL tracking
4. Build reporting.py for charts and summaries

Happy backtesting! 🚀
