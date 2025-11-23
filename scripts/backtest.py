#!/usr/bin/env python3
"""
Main Backtest Script

This script runs a backtest using the 200-day moving average (DMA) strategy
and displays formatted performance results.

Usage:
    python backtest.py SYMBOL [--start-date YYYY-MM-DD] [--end-date YYYY-MM-DD] [--slippage BPS] [--dollar-size SIZE]

Examples:
    python backtest.py AAPL
    python backtest.py AAPL --start-date 2020-01-01
    python backtest.py AAPL --start-date 2020-01-01 --end-date 2024-12-31
    python backtest.py AAPL --slippage 5 --dollar-size 250000
"""

import sys
from pathlib import Path
import argparse
from datetime import datetime
import logging

# Add parent directory to path to import libs
sys.path.insert(0, str(Path(__file__).parent.parent))

from strategies.dma import DMA
from strategies.vol_normalized_buy_and_hold import VolNormalizedBuyAndHold
from libs.backtester import Backtester


def parse_arguments():
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description='Run backtest using 200-day moving average strategy',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s AAPL
  %(prog)s AAPL --start-date 2020-01-01
  %(prog)s AAPL --start-date 2020-01-01 --end-date 2024-12-31
  %(prog)s AAPL --slippage 5 --dollar-size 250000
        """
    )

    parser.add_argument(
        'symbol',
        type=str,
        help='Stock ticker symbol (e.g., AAPL, MSFT, GOOGL)'
    )

    parser.add_argument(
        '--start-date',
        type=str,
        default='2010-01-01',
        help='Start date for backtest (YYYY-MM-DD) [default: 2010-01-01]'
    )

    parser.add_argument(
        '--end-date',
        type=str,
        default=datetime.today().strftime('%Y-%m-%d'),
        help='End date for backtest (YYYY-MM-DD) [default: today]'
    )

    parser.add_argument(
        '--slippage',
        type=float,
        default=5.0,
        help='Slippage in basis points [default: 5]'
    )

    parser.add_argument(
        '--dollar-size',
        type=float,
        default=100000,
        help='Dollar size for position sizing [default: 100000]'
    )

    parser.add_argument(
        '--lookback',
        type=int,
        default=200,
        help='Moving average lookback period [default: 200]'
    )

    parser.add_argument(
        '--verbose',
        '-v',
        action='store_true',
        help='Enable verbose logging'
    )

    return parser.parse_args()


def validate_date(date_string):
    """
    Validate date string format.

    Args:
        date_string (str): Date string to validate

    Returns:
        bool: True if valid, False otherwise
    """
    try:
        datetime.strptime(date_string, '%Y-%m-%d')
        return True
    except ValueError:
        return False


def display_performance_table(backtester, bnh_backtester):
    """
    Display performance metrics in a neat comparative table.

    Args:
        backtester: DMA Backtester instance with completed backtest
        bnh_backtester: Buy & Hold Backtester instance with completed backtest
    """
    unslipped = backtester.unslipped_performance
    slipped = backtester.slipped_performance
    bnh = bnh_backtester.slipped_performance

    # Define the metrics to display
    # Split into two groups: with and without trade-related metrics
    all_metrics = [
        ('Sharpe Ratio', 'sharpe', '.3f', False),
        ('Total PnL', 'total_pnl', ',.2f', False),
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
    print(f"PERFORMANCE COMPARISON: {backtester.symbol}")
    print(f"Period: {backtester.start_date} to {backtester.end_date}")
    print(f"Dollar Size: ${backtester.dollar_size:,.2f} | Slippage: {backtester.slippage_bps} bps")
    print("=" * 100)
    print()

    # Print table header
    header = f"{'Metric':<{metric_width}} {'Unslipped':>{value_width}} {'Slipped':>{value_width}} {'Buy & Hold':>{value_width}}"
    print(header)
    print("-" * len(header))

    # Print each metric
    for label, key, fmt, is_trade_metric in all_metrics:
        unslipped_val = unslipped.get(key, 0)
        slipped_val = slipped.get(key, 0)
        bnh_val = bnh.get(key, 0)

        # Format values based on the format string
        if 'f' in fmt:
            # Numeric formatting
            if key in ['total_pnl', 'mean_pnl_per_trade', 'avg_pnl_win', 'avg_pnl_loss',
                       'max_drawdown', 'drawdown_2', 'drawdown_3']:
                # Dollar amounts
                unslipped_str = f"${unslipped_val:{fmt}}"
                slipped_str = f"${slipped_val:{fmt}}"
                bnh_str = f"${bnh_val:{fmt}}" if not is_trade_metric else "-"
            else:
                # Non-dollar amounts
                unslipped_str = f"{unslipped_val:{fmt}}"
                slipped_str = f"{slipped_val:{fmt}}"
                bnh_str = f"{bnh_val:{fmt}}" if not is_trade_metric else "-"
        else:
            unslipped_str = str(unslipped_val)
            slipped_str = str(slipped_val)
            bnh_str = str(bnh_val) if not is_trade_metric else "-"

        print(f"{label:<{metric_width}} {unslipped_str:>{value_width}} {slipped_str:>{value_width}} {bnh_str:>{value_width}}")

    print("=" * 100)

    # Print comparison summary
    print()
    print("STRATEGY COMPARISON (Slipped DMA vs Buy & Hold):")
    print("-" * 100)

    pnl_diff = slipped['total_pnl'] - bnh['total_pnl']
    pnl_diff_pct = (pnl_diff / abs(bnh['total_pnl']) * 100) if bnh['total_pnl'] != 0 else 0
    sharpe_diff = slipped['sharpe'] - bnh['sharpe']

    print(f"  PnL Difference:     ${pnl_diff:+,.2f} ({pnl_diff_pct:+.2f}%)")
    print(f"  Sharpe Difference:  {sharpe_diff:+.3f}")

    if pnl_diff > 0:
        print(f"  → DMA outperformed Buy & Hold")
    elif pnl_diff < 0:
        print(f"  → Buy & Hold outperformed DMA")
    else:
        print(f"  → Identical performance")

    print("=" * 100)


def main():
    """
    Main function to run the backtest.
    """
    # Parse command-line arguments
    args = parse_arguments()

    # Configure logging
    log_level = logging.INFO if args.verbose else logging.WARNING
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Validate dates
    if not validate_date(args.start_date):
        print(f"Error: Invalid start date format: {args.start_date}")
        print("Expected format: YYYY-MM-DD")
        sys.exit(1)

    if not validate_date(args.end_date):
        print(f"Error: Invalid end date format: {args.end_date}")
        print("Expected format: YYYY-MM-DD")
        sys.exit(1)

    # Print header
    print()
    print("=" * 80)
    print("QUANTSTRAT BACKTEST")
    print("=" * 80)
    print(f"Symbol:         {args.symbol}")
    print(f"Strategy:       {args.lookback}-Day Moving Average (DMA)")
    print(f"Period:         {args.start_date} to {args.end_date}")
    print(f"Dollar Size:    ${args.dollar_size:,.2f}")
    print(f"Slippage:       {args.slippage} bps")
    print("=" * 80)
    print()

    try:
        # Create DMA strategy instance
        strategy = DMA(lookback=args.lookback)

        # Create backtester instance
        backtester = Backtester(strategy, dollar_size=args.dollar_size)

        # Run DMA backtest
        print("Running DMA backtest...")
        backtester(
            symbol=args.symbol,
            start_date=args.start_date,
            end_date=args.end_date,
            slippage_bps=args.slippage
        )

        # Check if backtest was successful
        if backtester.slipped_performance is None:
            print()
            print("=" * 80)
            print("ERROR: DMA backtest failed")
            print("=" * 80)
            print()
            print("Possible reasons:")
            print("  - Invalid ticker symbol")
            print("  - No data available for the specified date range")
            print("  - Network connectivity issues")
            print()
            sys.exit(1)

        # Run Buy & Hold backtest for comparison
        print("Running Buy & Hold backtest...")
        bnh_strategy = VolNormalizedBuyAndHold()
        bnh_backtester = Backtester(bnh_strategy, dollar_size=args.dollar_size)
        bnh_backtester(
            symbol=args.symbol,
            start_date=args.start_date,
            end_date=args.end_date,
            slippage_bps=args.slippage
        )

        if bnh_backtester.slipped_performance is None:
            print()
            print("WARNING: Buy & Hold backtest failed, showing DMA results only")
            print()
            # Fall back to single-strategy display (would need original function)
            sys.exit(1)

        # Display results in a neat comparative table
        print()
        display_performance_table(backtester, bnh_backtester)
        print()

        # Create visualization with Buy & Hold comparison
        strategy_name = args.lookback
        output_filename = f"output/{args.symbol}-{strategy_name}dma.png"
        print(f"Saving visualization to {output_filename}...")
        backtester.visualize(output_filename, bnh_backtester=bnh_backtester)
        print(f"Visualization saved successfully!")
        print()

    except KeyboardInterrupt:
        print()
        print("=" * 80)
        print("Backtest interrupted by user")
        print("=" * 80)
        print()
        sys.exit(1)

    except Exception as e:
        print()
        print("=" * 80)
        print(f"ERROR: {str(e)}")
        print("=" * 80)
        print()
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
