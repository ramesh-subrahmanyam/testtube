#!/usr/bin/env python3
"""
Portfolio Backtest Script

This script runs a backtest across multiple symbols using a specified strategy
and displays aggregated portfolio performance results.

Usage:
    python portfolio_backtest.py SYMBOLS_FILE [--start-date YYYY-MM-DD] [--end-date YYYY-MM-DD] [--slippage BPS] [--dollar-size SIZE]

Examples:
    python portfolio_backtest.py symbols.txt
    python portfolio_backtest.py symbols.txt --start-date 2020-01-01
    python portfolio_backtest.py symbols.txt --start-date 2020-01-01 --end-date 2024-12-31
    python portfolio_backtest.py symbols.txt --slippage 5 --dollar-size 250000

Symbols File Format:
    One symbol per line:
    AAPL
    MSFT
    GOOGL
    TSLA
"""

import sys
from pathlib import Path
import argparse
from datetime import datetime
import logging

# Add parent directory to path to import libs
sys.path.insert(0, str(Path(__file__).parent.parent))

from strategies.dma import DMA
from libs.portfolio import PortfolioBacktester


def parse_arguments():
    """
    Parse command-line arguments for the new configuration-based portfolio backtest.

    Returns:
        argparse.Namespace: Parsed arguments
    """
    import argparse
    parser = argparse.ArgumentParser(
        description='Run a portfolio backtest across multiple symbols using a configuration file.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --config config.yaml --symbols symbols.txt --start 2020-01-01 --end 2024-12-31
  %(prog)s --config config.json --symbols symbols.txt --start 2020-01-01 --end 2024-12-31 --slippage 5

Symbols File Format:
  One symbol per line (empty lines and lines starting with # are ignored):
  AAPL
  MSFT
  GOOGL
  # This is a comment
  TSLA
        """
    )
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to the configuration file (YAML or JSON)'
    )
    parser.add_argument(
        '--symbols',
        type=str,
        required=True,
        help='Path to the file containing symbols (one per line)'
    )
    parser.add_argument(
        '--start',
        type=str,
        required=True,
        help='Start date (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--end',
        type=str,
        required=True,
        help='End date (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--slippage',
        type=float,
        default=0.0,
        help='Slippage in basis points (default: 0)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Optional path to save the aggregated results CSV'
    )
    parser.add_argument(
        '--plot',
        type=str,
        default=None,
        help='Optional path to save the performance plot'
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


def read_symbols_file(file_path):
    """
    Read symbols from a file.

    Args:
        file_path (str): Path to symbols file

    Returns:
        list: List of symbols

    The file should contain one symbol per line.
    Empty lines and lines starting with # are ignored.
    """
    symbols = []

    try:
        with open(file_path, 'r') as f:
            for line in f:
                # Strip whitespace
                line = line.strip()

                # Skip empty lines and comments
                if not line or line.startswith('#'):
                    continue

                # Add symbol (convert to uppercase)
                symbols.append(line.upper())

    except FileNotFoundError:
        print(f"Error: Symbols file not found: {file_path}")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading symbols file: {str(e)}")
        sys.exit(1)

    return symbols


def display_performance_table(portfolio):
    """
    Display portfolio performance metrics in a neat table.

    Args:
        portfolio: PortfolioBacktester instance with completed backtest
    """
    port_perf = portfolio.portfolio_performance

    # Define the metrics to display
    metrics = [
        ('Sharpe Ratio', 'sharpe', '.3f'),
        ('Sharpe (Exposure)', 'sharpe_exposure', '.3f'),
        ('Total PnL', 'total_pnl', ',.2f'),
        ('#Days of Exposure', 'num_exposure_days', ',.0f'),
        ('PnL per Exposure Day', 'pnl_per_exposure_day', ',.2f'),
        ('Number of Trades', 'num_trades', ',.0f'),
        ('Mean PnL per Trade', 'mean_pnl_per_trade', ',.2f'),
        ('#Wins', 'num_wins', ',.0f'),
        ('Average PnL/Win', 'avg_pnl_win', ',.2f'),
        ('#Losses', 'num_losses', ',.0f'),
        ('Average PnL/Loss', 'avg_pnl_loss', ',.2f'),
        ('Max Drawdown', 'max_drawdown', ',.2f'),
        ('Drawdown #2', 'drawdown_2', ',.2f'),
        ('Drawdown #3', 'drawdown_3', ',.2f'),
    ]

    # Calculate column widths
    metric_width = max(len(m[0]) for m in metrics) + 2
    value_width = 20

    # Print header
    print("=" * 80)
    print("PORTFOLIO PERFORMANCE")
    print(f"Symbols: {', '.join(portfolio.symbols)}")
    print(f"Period: {portfolio.start_date} to {portfolio.end_date}")
    print(f"Dollar Size/Symbol: ${portfolio.dollar_size:,.2f} | Slippage: {portfolio.slippage_bps} bps")
    print("=" * 80)
    print()

    # Print individual symbol summary
    print("INDIVIDUAL SYMBOL SUMMARY:")
    print("-" * 80)
    print(f"{'Symbol':<10} {'Total PnL':>15} {'Sharpe':>10} {'#Trades':>10}")
    print("-" * 80)

    for symbol in portfolio.symbols:
        bt = portfolio.backtesters[symbol]
        perf = bt.slipped_performance
        print(f"{symbol:<10} ${perf['total_pnl']:>14,.2f} {perf['sharpe']:>10.3f} {perf['num_trades']:>10,}")

    print()
    print("PORTFOLIO AGGREGATED METRICS:")
    print("-" * 80)

    # Print each metric
    for label, key, fmt in metrics:
        value = port_perf.get(key, 0)

        # Format values based on the format string
        if 'f' in fmt:
            # Numeric formatting
            if key in ['total_pnl', 'mean_pnl_per_trade', 'avg_pnl_win', 'avg_pnl_loss',
                       'max_drawdown', 'drawdown_2', 'drawdown_3', 'pnl_per_exposure_day']:
                # Dollar amounts
                value_str = f"${value:{fmt}}"
            else:
                # Non-dollar amounts
                value_str = f"{value:{fmt}}"
        else:
            value_str = str(value)

        print(f"{label:<{metric_width}} {value_str:>{value_width}}")

    print("=" * 80)


def main():
    """
    Main function to run the portfolio backtest using configuration file.
    """
    import sys
    import os
    # Add the scripts directory to the path to import from backtest
    scripts_dir = os.path.dirname(os.path.abspath(__file__))
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    
    try:
        from backtest import load_config, instantiate_strategy, instantiate_exposure_manager
    except ImportError as e:
        print(f"Error importing from backtest: {e}")
        print("Make sure backtest.py is in the same directory (scripts/)")
        sys.exit(1)
    
    from strategies import get_strategy_class, STRATEGY_REGISTRY
    
    # Parse command-line arguments
    args = parse_arguments()

    # Configure logging
    log_level = logging.INFO if args.verbose else logging.WARNING
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Validate dates
    if not validate_date(args.start):
        print(f"Error: Invalid start date format: {args.start}")
        print("Expected format: YYYY-MM-DD")
        sys.exit(1)

    if not validate_date(args.end):
        print(f"Error: Invalid end date format: {args.end}")
        print("Expected format: YYYY-MM-DD")
        sys.exit(1)

    # Load configuration
    config = load_config(args.config)
    strategy_config = config.get('strategy', {})
    exposure_config = config.get('exposure_management', None)

    # Get strategy class and parameters
    strategy_name = strategy_config.get('name')
    if not strategy_name:
        raise ValueError("Configuration must specify a strategy name under 'strategy.name'")
    try:
        strategy_class = get_strategy_class(strategy_name)
    except ValueError as e:
        print(f"Error: {e}")
        print("Available strategies:", ', '.join(STRATEGY_REGISTRY.keys()))
        sys.exit(1)
    strategy_params = strategy_config.get('params', {})

    # Instantiate exposure manager if provided
    exposure_manager = None
    if exposure_config:
        try:
            exposure_manager = instantiate_exposure_manager(exposure_config)
        except Exception as e:
            print(f"Warning: Failed to instantiate exposure manager: {e}")
            print("Continuing without exposure manager.")

    # Read symbols
    symbols = read_symbols_file(args.symbols)

    if not symbols:
        print("Error: No symbols found in file")
        sys.exit(1)

    # Create portfolio backtester
    dollar_size = config.get('dollar_size', 100000)
    
    # Print header
    print()
    print("=" * 80)
    print("QUANTSTRAT PORTFOLIO BACKTEST")
    print("=" * 80)
    print(f"Symbols:        {', '.join(symbols)} ({len(symbols)} total)")
    print(f"Strategy:       {strategy_name} with params {strategy_params}")
    print(f"Period:         {args.start} to {args.end}")
    print(f"Dollar Size:    ${dollar_size:,.2f} per symbol")
    print(f"Slippage:       {args.slippage} bps")
    if exposure_config:
        print(f"Exposure Config: {exposure_config}")
    if exposure_manager:
        print(f"Exposure Manager: {exposure_manager.__class__.__name__}")
    print("=" * 80)
    print()

    try:
        # Create portfolio backtester with exposure manager if provided
        portfolio = PortfolioBacktester(
            strategy_class=strategy_class,
            strategy_params=strategy_params,
            dollar_size=dollar_size,
            exposure_manager=exposure_manager
        )

        # Run portfolio backtest
        print("Running portfolio backtest...")
        portfolio(
            symbols=symbols,
            start_date=args.start,
            end_date=args.end,
            slippage_bps=args.slippage
        )

        # Check if backtest was successful
        if portfolio.portfolio_performance is None:
            print()
            print("=" * 80)
            print("ERROR: Portfolio backtest failed")
            print("=" * 80)
            print()
            print("Possible reasons:")
            print("  - All symbols failed to backtest")
            print("  - No data available for the specified date range")
            print("  - Network connectivity issues")
            print()
            sys.exit(1)

        # Display results
        print()
        display_performance_table(portfolio)
        print()

        # Save output if requested
        if args.output:
            df = portfolio.get_dataframe()
            df.to_csv(args.output, index=False)
            print(f"Results saved to {args.output}")

        # Generate plot if requested
        if args.plot:
            portfolio.visualize(args.plot)
            print(f"Plot saved to {args.plot}")
        else:
            # Create default visualization
            # Ensure output directory exists
            output_dir = "output"
            os.makedirs(output_dir, exist_ok=True)
            output_filename = f"{output_dir}/portfolio-{strategy_name.lower()}.png"
            print(f"Saving visualization to {output_filename}...")
            portfolio.visualize(output_filename)
            print(f"Visualization saved successfully!")
            print()

    except KeyboardInterrupt:
        print()
        print("=" * 80)
        print("Portfolio backtest interrupted by user")
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
