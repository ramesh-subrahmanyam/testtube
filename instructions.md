I want to implement a backtesting engine for daily price series.  
This will fetch stock prices, run a strategy to determine daily positions and present performance results at an individula stock level and a portfoliko level.

I want to structure it as follows:
1. a module to fetch time series data from yahoo finance and cache it.  Name it libs/prices.py.  It should have functions to fetch stock prices (reuse fetch_stock_prices from ../optionspnl/libs/dma_analysis.py).  It should read from the prices cache if it is not stale and should refetch and update the cache if it is stale.  

2. Implement a module libs/cache.py that will cache the prices.  There should be a refresh_cache method to firce refresh pf the cache.  There should also be cache configuration parameters (days to go stale etc.).  I suggeest you store each stock's data in a separate parquet file.  

Questions: 
- Should staleness be stock specific? 
- Can you reuse/refactor any ofthe caching code in optionspnl/libs?  - How will cache configuration be implemented?  This is a library, but will the top-level script need to pas it down?  How? parameters? json?

3.  Module for specific strategies.  Create an abstract base class in strategies/base.py.  It should have an __init__ method which will accept the volatility_function and strategy parameters (strategy-specific).  It should have a __call__ method which will take a symbol and start_date, end_date.  It will read the prices using libs/prices.py and create a positions series.   It will also calculate a rolling volatility series using the volatility calculation function.  It will store the prices, volatility (rolling)  and positions series in attribute self.df

4.  There will be a libs/volatility.py  module.  This will start with one volatility function simple_vol which computes the daily return (ie today's cose/yesterday's close -1) and compute the trailing N-day standard deviation.  N is a parameter which should be partialed out -- so that simople_vol(20) means 20-day std-dev of 1-day returns

5. Implement the module signals/technical.py with a function SMA that computes the N-day simple moving average over N days.  Use decorators so that I can refer to SMA(200) to compute a 200-day moving average.  The signal output should be NaN for the first N days since there is not enough data to compute the SMA.

6. base_strategy.py should exclude the period corresponding to the initial string of NaNs in the signal (some signals have a warm-up period).  If there are NaNs in the middle of the signal series the strategy should fail.  It should create an err file in teh currenyt directory.  The err file should say that there were NaNs in the moddl of the signal (name the signal function).  It should describe how many such NaNs and list the dates.

7. Implement module strategies.dma.  Define a strategy clas deriuved from strategies.base.Strategy. Use the SMA signal to determine positions.  IF pruce >= sma, position is 1; otherwise 0.

8. Implementlibs/backtester.py.  This will have a class Backtester.  the __init__ method will take as inputs the Strategy object and a dollar size.  The __call__method will take the symbol, start_date and end_date, and slippage_bps (slippage in basis points).  It will run the strategy on the symbol and the start/end dates.  It will read the position and price series from the Strategy.df dataframe.  The dollar volatility will be calculated as (return) volatility times price.  The positions will be vol_adjusted (ie multiplied by dollar_size/dollar_volatility). The unslipped daily pnl will be calculated.  Slippage will be estimated on each day as abs(change in position from previous day) * today's price * slippage_bps/100.  Subtract slippage (series)from pnl series to give slipped_pnl series.  All computations should be vectorized for efficiency.  The stats() function should be invoked on the slipped and unslipped pnl series ans stored in slipped_perfformance an d unslipped_performance attributes.

9.  Implement a libs/performance.py module.  There should be a function stats(pnl_series) which should take the onl series as input and output a dictionary sharpe, total pnl, number of tradesa and mean pnl per trade

10. Create a backtest script with a __main__ program.  It should take optional args start_date (default=2010-01-01) and end_date (today).  the symbol will be given as argument.  It should run a backtest using the 200-dma strategy and neatly format the performance ad display on stad out.  Rmove all the demo/ecxampl files from scripts

11. Create a document DESIGN.md that describes the design and the main workflow (the sequence of method calls to get the work done).

