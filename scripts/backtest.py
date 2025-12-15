#!/usr/bin/env python3
"""
Main Backtest Script

This script runs a backtest using configuration from a JSON file.
The config file specifies the strategy, parameters, exposure management, and other settings.

Usage:
    python backtest.py SYMBOL [CONFIG_FILE]
    python backtest.py SYMBOL [--config CONFIG_FILE]

Examples:
    python backtest.py AAPL                              # Uses data/backtest.json with AAPL
    python backtest.py MSFT data/my_config.json          # MSFT with custom config
    python backtest.py SPY --config my_custom_config.json  # SPY with named config arg
"""

import sys
from pathlib import Path
import argparse
import json
import logging
from datetime import datetime

# Add parent directory to path to import libs
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import registries
from strategies import get_strategy_class
from libs.exposure_management import get_exposure_manager_class


def load_config(config_path):
    """
    Load configuration from JSON file.

    Args:
        config_path (str): Path to JSON config file

    Returns:
        dict: Configuration dictionary
    """
    import os
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    # Check if file is empty
    if os.path.getsize(config_path) == 0:
        raise ValueError(f"Configuration file is empty: {config_path}")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def instantiate_strategy(strategy_config):
    """
    Instantiate a strategy from configuration.

    Args:
        strategy_config (dict): Strategy configuration with 'name', 'params', 'signal_config', and 'entry_exit_config'

    Returns:
        BaseStrategy: Instantiated strategy object
    """
    strategy_class = get_strategy_class(strategy_config['name'])
    strategy = strategy_class(**strategy_config.get('params', {}))

    # Apply entry/exit timing configuration if present
    entry_exit_config = strategy_config.get('entry_exit_config', {})

    # Support two formats: old format (booleans) and new format (entry_mode string)
    entry_mode = entry_exit_config.get('entry_mode', None)
    enter_at_open = entry_exit_config.get('enter_at_open', False)
    enter_at_next_close = entry_exit_config.get('enter_at_next_close', False)
    exit_at_open = entry_exit_config.get('exit_at_open', False)

    # If entry_mode is specified, use it (new format)
    if entry_mode:
        entry_mode = entry_mode.lower()
        if entry_mode == 'open' or entry_mode == 'next_open':
            strategy = strategy.enter_at_next_open()
        elif entry_mode == 'next_close':
            strategy = strategy.enter_at_next_close()
        elif entry_mode == 'close':
            pass  # Default behavior
        else:
            raise ValueError(f"Invalid entry_mode: {entry_mode}. Must be 'close', 'open', 'next_open', or 'next_close'")
    else:
        # Old format: use boolean flags
        if enter_at_open:
            strategy = strategy.enter_at_next_open()
        if enter_at_next_close:
            strategy = strategy.enter_at_next_close()

    if exit_at_open:
        strategy = strategy.exit_at_next_open()

    # Apply signal configuration if present
    signal_config = strategy_config.get('signal_config', {})
    frequency = signal_config.get('frequency', 'daily')

    if frequency == 'weekly':
        day = signal_config.get('day', 'monday')
        strategy = strategy.periodic('weekly', day=day)
    elif frequency == 'monthly':
        when = signal_config.get('when', 'end')
        strategy = strategy.periodic('monthly', when=when)
    # 'daily' or unspecified means no periodic modification needed

    return strategy


def instantiate_exposure_manager(exp_config):
    """
    Instantiate an exposure manager from configuration.

    Args:
        exp_config (dict): Exposure management configuration

    Returns:
        ExposureManager: Instantiated exposure manager object
    """
    exp_class = get_exposure_manager_class(exp_config['name'])
    exp_manager = exp_class(**exp_config.get('params', {}))

    # Apply rebalance configuration if present
    rebalance_config = exp_config.get('rebalance', {})
    if rebalance_config:
        frequency = rebalance_config.get('frequency', 'monthly')
        when = rebalance_config.get('when', 'start')
        exp_manager = exp_manager.periodic(frequency=frequency, when=when)

    return exp_manager


def parse_arguments():
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description='Run backtest using JSON configuration file',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s AAPL                              # Uses data/backtest.json with AAPL
  %(prog)s MSFT data/my_config.json          # MSFT with custom config
  %(prog)s SPY --config my_custom_config.json  # SPY with named config arg
        """
    )

    parser.add_argument(
        'symbol',
        type=str,
        help='Stock symbol to backtest (e.g., AAPL, MSFT, SPY)'
    )

    parser.add_argument(
        'config_file',
        nargs='?',
        type=str,
        default='data/backtest.json',
        help='Path to JSON configuration file [default: data/backtest.json]'
    )

    parser.add_argument(
        '--config',
        type=str,
        dest='config_override',
        help='Alternative way to specify config file path (overrides positional argument)'
    )

    args = parser.parse_args()

    # If --config is provided, it takes precedence over positional argument
    if args.config_override:
        args.config = args.config_override
    else:
        args.config = args.config_file

    return args


def display_performance_table(backtester, bnh_backtester):
    """
    Display performance metrics in a neat comparative table.

    Args:
        backtester: Strategy Backtester instance with completed backtest
        bnh_backtester: Buy & Hold Backtester instance with completed backtest
    """
    unslipped = backtester.unslipped_performance
    slipped = backtester.slipped_performance
    bnh = bnh_backtester.slipped_performance if bnh_backtester else None

    # Define the metrics to display
    all_metrics = [
        ('Sharpe Ratio', 'sharpe', '.3f', False),
        ('Sharpe (Exposure)', 'sharpe_exposure', '.3f', False),
        ('Total PnL', 'total_pnl', ',.2f', False),
        ('#Days of Exposure', 'num_exposure_days', ',.0f', False),
        ('PnL per Exposure Day', 'pnl_per_exposure_day', ',.2f', False),
        ('Number of Trades', 'num_trades', ',.0f', True),
        ('Mean PnL per Trade', 'mean_pnl_per_trade', ',.2f', True),
        ('#Wins', 'num_wins', ',.0f', True),
        ('Average PnL/Win', 'avg_pnl_win', ',.2f', True),
        ('Days Held (Wins)', 'days_held_wins', '.1f', True),
        ('#Losses', 'num_losses', ',.0f', True),
        ('Average PnL/Loss', 'avg_pnl_loss', ',.2f', True),
        ('Days Held (Loss)', 'days_held_losses', '.1f', True),
        ('Max Drawdown', 'max_drawdown', ',.2f', False),
        ('Drawdown #2', 'drawdown_2', ',.2f', False),
        ('Drawdown #3', 'drawdown_3', ',.2f', False),
    ]

    # Calculate column widths
    metric_width = max(len(m[0]) for m in all_metrics) + 2
    value_width = 18

    # Print header
    print("=" * 100)
    print(f"PERFORMANCE RESULTS")
    print("=" * 100)
    print()

    # Print table header
    if bnh:
        header = f"{'Metric':<{metric_width}} {'Unslipped':>{value_width}} {'Slipped':>{value_width}} {'Buy & Hold':>{value_width}}"
    else:
        header = f"{'Metric':<{metric_width}} {'Unslipped':>{value_width}} {'Slipped':>{value_width}}"
    print(header)
    print("-" * len(header))

    # Print each metric
    for label, key, fmt, is_trade_metric in all_metrics:
        unslipped_val = unslipped.get(key, 0)
        slipped_val = slipped.get(key, 0)

        # Format values based on the format string
        if 'f' in fmt:
            # Numeric formatting
            if key in ['total_pnl', 'mean_pnl_per_trade', 'avg_pnl_win', 'avg_pnl_loss',
                       'max_drawdown', 'drawdown_2', 'drawdown_3', 'pnl_per_exposure_day']:
                # Dollar amounts
                unslipped_str = f"${unslipped_val:{fmt}}"
                slipped_str = f"${slipped_val:{fmt}}"
            else:
                # Non-dollar amounts
                unslipped_str = f"{unslipped_val:{fmt}}"
                slipped_str = f"{slipped_val:{fmt}}"
        else:
            unslipped_str = str(unslipped_val)
            slipped_str = str(slipped_val)

        if bnh:
            bnh_val = bnh.get(key, 0)
            if 'f' in fmt:
                if key in ['total_pnl', 'mean_pnl_per_trade', 'avg_pnl_win', 'avg_pnl_loss',
                           'max_drawdown', 'drawdown_2', 'drawdown_3', 'pnl_per_exposure_day']:
                    bnh_str = f"${bnh_val:{fmt}}" if not is_trade_metric else "-"
                else:
                    bnh_str = f"{bnh_val:{fmt}}" if not is_trade_metric else "-"
            else:
                bnh_str = str(bnh_val) if not is_trade_metric else "-"
            print(f"{label:<{metric_width}} {unslipped_str:>{value_width}} {slipped_str:>{value_width}} {bnh_str:>{value_width}}")
        else:
            print(f"{label:<{metric_width}} {unslipped_str:>{value_width}} {slipped_str:>{value_width}}")

    print("=" * 100)

    # Print comparison summary if benchmark exists
    if bnh:
        print()
        print("STRATEGY COMPARISON (Slipped Strategy vs Buy & Hold):")
        print("-" * 100)

        pnl_diff = slipped['total_pnl'] - bnh['total_pnl']
        pnl_diff_pct = (pnl_diff / abs(bnh['total_pnl']) * 100) if bnh['total_pnl'] != 0 else 0
        sharpe_diff = slipped['sharpe'] - bnh['sharpe']

        print(f"  PnL Difference:     ${pnl_diff:+,.2f} ({pnl_diff_pct:+.2f}%)")
        print(f"  Sharpe Difference:  {sharpe_diff:+.3f}")

        if pnl_diff > 0:
            print(f"  → Strategy outperformed Buy & Hold")
        elif pnl_diff < 0:
            print(f"  → Buy & Hold outperformed Strategy")
        else:
            print(f"  → Identical performance")

        print("=" * 100)


def main():
    """
    Main function to run the backtest.
    """
    # Parse command-line arguments
    args = parse_arguments()

    # Load configuration
    try:
        config = load_config(args.config)
    except FileNotFoundError:
        print(f"Error: Configuration file not found: {args.config}")
        print(f"Please create a configuration file or specify a valid path.")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in configuration file: {args.config}")
        print(f"Details: {str(e)}")
        sys.exit(1)

    # Extract configuration sections
    backtest_config = config.get('backtest', {})
    strategy_config = config.get('strategy', {})
    exp_mgmt_config = config.get('exposure_management', {})
    benchmark_config = config.get('benchmark', {})
    output_config = config.get('output', {})

    # Configure logging
    log_level = logging.INFO if backtest_config.get('verbose', False) else logging.ERROR
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Print header
    print()
    print("=" * 100)
    print("BACKTEST CONFIGURATION")
    print("=" * 100)
    print(f"Config File:            {args.config}")
    print(f"Symbol:                 {args.symbol}")
    print(f"Period:                 {backtest_config.get('start_date', 'N/A')} to {backtest_config.get('end_date', 'N/A')}")
    print(f"Strategy:               {strategy_config.get('name', 'N/A')}")
    print(f"Strategy Params:        {strategy_config.get('params', {})}")

    # Display entry/exit timing configuration if present
    entry_exit_config = strategy_config.get('entry_exit_config', {})
    if entry_exit_config:
        # Determine entry timing
        entry_mode = entry_exit_config.get('entry_mode', None)
        if entry_mode:
            entry_mode_lower = entry_mode.lower()
            if entry_mode_lower == 'open' or entry_mode_lower == 'next_open':
                entry_timing = "Next Open"
            elif entry_mode_lower == 'next_close':
                entry_timing = "Next Close (lag=1)"
            else:
                entry_timing = "Close"
        else:
            # Old format
            enter_at_open = entry_exit_config.get('enter_at_open', False)
            enter_at_next_close = entry_exit_config.get('enter_at_next_close', False)
            if enter_at_open:
                entry_timing = "Next Open"
            elif enter_at_next_close:
                entry_timing = "Next Close (lag=1)"
            else:
                entry_timing = "Close"

        exit_at_open = entry_exit_config.get('exit_at_open', False)
        exit_timing = "Next Open" if exit_at_open else "Close"
        print(f"Entry/Exit Timing:      Entry at {entry_timing}, Exit at {exit_timing}")

    # Display signal configuration if present
    signal_config = strategy_config.get('signal_config', {})
    if signal_config:
        freq = signal_config.get('frequency', 'daily')
        if freq == 'weekly':
            day = signal_config.get('day', 'monday')
            print(f"Signal Frequency:       Weekly ({day.capitalize()})")
        elif freq == 'monthly':
            when = signal_config.get('when', 'end')
            print(f"Signal Frequency:       Monthly ({when.capitalize()})")
        else:
            print(f"Signal Frequency:       {freq.capitalize()}")

    try:
        # Instantiate strategy
        strategy = instantiate_strategy(strategy_config)

        # Instantiate exposure manager
        exp_manager = instantiate_exposure_manager(exp_mgmt_config)

        # Print exposure management info
        print(f"Exposure Management:    {exp_mgmt_config.get('name', 'N/A')}")
        print(f"Exposure Params:        {exp_mgmt_config.get('params', {})}")
        rebalance_config = exp_mgmt_config.get('rebalance', {})
        if rebalance_config:
            freq = rebalance_config.get('frequency', 'N/A')
            when = rebalance_config.get('when', 'N/A')
            print(f"Rebalance:              {freq.capitalize()} ({when})")
        print(f"Slippage:               {backtest_config.get('slippage_bps', 5.0)} bps")
        print("=" * 100)
        print()

        # Import Backtester here to avoid circular imports
        from libs.backtester import Backtester

        # Create backtester instance
        backtester = Backtester(strategy, exposure_manager=exp_manager)

        # Run backtest (use symbol from command line)
        backtester(
            symbol=args.symbol,
            start_date=backtest_config.get('start_date', '2010-01-01'),
            end_date=backtest_config.get('end_date', datetime.today().strftime('%Y-%m-%d')),
            slippage_bps=backtest_config.get('slippage_bps', 5.0)
        )

        # Check if backtest was successful
        if backtester.slipped_performance is None:
            print()
            print("=" * 80)
            print("ERROR: Backtest failed")
            print("=" * 80)
            print()
            print("Possible reasons:")
            print("  - Invalid ticker symbol")
            print("  - No data available for the specified date range")
            print("  - Network connectivity issues")
            print()
            sys.exit(1)

        # Run benchmark if enabled
        bnh_backtester = None
        if benchmark_config.get('enabled', True):
            print("Running benchmark backtest...")
            bnh_strategy = instantiate_strategy(benchmark_config)

            # Use same exposure management if configured
            if benchmark_config.get('use_same_exposure_management', True):
                bnh_exp_mgr = instantiate_exposure_manager(exp_mgmt_config)
            else:
                # Use benchmark-specific exposure management if provided
                bnh_exp_config = benchmark_config.get('exposure_management', exp_mgmt_config)
                bnh_exp_mgr = instantiate_exposure_manager(bnh_exp_config)

            bnh_backtester = Backtester(bnh_strategy, exposure_manager=bnh_exp_mgr)
            bnh_backtester(
                symbol=args.symbol,
                start_date=backtest_config.get('start_date', '2010-01-01'),
                end_date=backtest_config.get('end_date', datetime.today().strftime('%Y-%m-%d')),
                slippage_bps=backtest_config.get('slippage_bps', 5.0)
            )

            if bnh_backtester.slipped_performance is None:
                print()
                print("WARNING: Benchmark backtest failed, showing strategy results only")
                print()
                bnh_backtester = None

        # Display results
        print()
        display_performance_table(backtester, bnh_backtester)
        print()

        # Create visualization if enabled
        if output_config.get('visualization', {}).get('enabled', True):
            output_path = output_config.get('visualization', {}).get('output_path',
                                                                      'output/{symbol}-{strategy_name}.png')
            # Format the output path
            output_filename = output_path.format(
                symbol=args.symbol,
                strategy_name=strategy_config.get('name', 'strategy')
            )
            print(f"Saving visualization to {output_filename}...")
            backtester.visualize(output_filename, bnh_backtester=bnh_backtester)
            print(f"Visualization saved successfully!")
            print()

    except KeyError as e:
        print()
        print("=" * 80)
        print(f"ERROR: Missing required configuration key: {str(e)}")
        print("=" * 80)
        print()
        print("Please check your configuration file for completeness.")
        sys.exit(1)

    except Exception as e:
        print()
        print("=" * 80)
        print(f"ERROR: {str(e)}")
        print("=" * 80)
        print()
        if backtest_config.get('verbose', False):
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
