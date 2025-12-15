#!/usr/bin/env python3
"""
Strategy Comparison Script

This script compares multiple strategies (e.g., DMA vs Buy & Hold) and displays
side-by-side performance metrics.

Usage:
    python compare_strategies.py SYMBOL [--start-date YYYY-MM-DD] [--end-date YYYY-MM-DD] [--slippage BPS] [--dollar-size SIZE]

Examples:
    python compare_strategies.py AAPL
    python compare_strategies.py AAPL --start-date 2020-01-01
    python compare_strategies.py AAPL --start-date 2020-01-01 --end-date 2024-12-31
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
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Compare DMA strategy vs Vol-Normalized Buy & Hold',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s AAPL
  %(prog)s AAPL --start-date 2020-01-01
  %(prog)s AAPL --start-date 2020-01-01 --end-date 2024-12-31
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
        help='DMA lookback period [default: 200]'
    )

    parser.add_argument(
        '--verbose',
        '-v',
        action='store_true',
        help='Enable verbose logging'
    )

    return parser.parse_args()


def create_dual_axis_plot(dma_bt, bnh_bt, args, output_path):
    """
    Create a dual-axis plot comparing DMA and Buy & Hold strategies.

    Args:
        dma_bt: DMA backtester instance
        bnh_bt: Buy & Hold backtester instance
        args: Command-line arguments
        output_path: Path to save the plot
    """
    try:
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        import matplotlib.pyplot as plt
    except ImportError as e:
        print(f"ERROR: matplotlib is required for visualization. Install with: pip install matplotlib ({e})")
        return

    dma_df = dma_bt.strategy.df
    bnh_df = bnh_bt.strategy.df
    dma_perf = dma_bt.slipped_performance
    bnh_perf = bnh_bt.slipped_performance

    # Create figure with dual y-axes
    fig, ax1 = plt.subplots(figsize=(14, 8))

    # Plot DMA on left y-axis
    color1 = 'tab:blue'
    ax1.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax1.set_ylabel('DMA Strategy Cumulative P&L ($)', fontsize=12, color=color1, fontweight='bold')
    line1 = ax1.plot(dma_df.index, dma_df['Cumulative_Slipped_PnL'],
                     label=f'{args.lookback}-DMA (Sharpe: {dma_perf["sharpe"]:.3f})',
                     linewidth=2.5, color=color1)
    ax1.tick_params(axis='y', labelcolor=color1, labelsize=10)
    ax1.grid(True, alpha=0.3, linestyle=':')

    # Set y-limits for left axis to use full scale
    dma_min = dma_df['Cumulative_Slipped_PnL'].min()
    dma_max = dma_df['Cumulative_Slipped_PnL'].max()
    dma_range = dma_max - dma_min
    ax1.set_ylim(dma_min - 0.05 * dma_range, dma_max + 0.05 * dma_range)

    # Format left y-axis as currency
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

    # Create second y-axis for Buy & Hold
    ax2 = ax1.twinx()
    color2 = 'tab:green'
    ax2.set_ylabel('Buy & Hold Cumulative P&L ($)', fontsize=12, color=color2, fontweight='bold')
    line2 = ax2.plot(bnh_df.index, bnh_df['Cumulative_Slipped_PnL'],
                     label=f'Buy & Hold (Sharpe: {bnh_perf["sharpe"]:.3f})',
                     linewidth=2.5, color=color2, alpha=0.8)
    ax2.tick_params(axis='y', labelcolor=color2, labelsize=10)

    # Set y-limits for right axis to use full scale
    bnh_min = bnh_df['Cumulative_Slipped_PnL'].min()
    bnh_max = bnh_df['Cumulative_Slipped_PnL'].max()
    bnh_range = bnh_max - bnh_min
    ax2.set_ylim(bnh_min - 0.05 * bnh_range, bnh_max + 0.05 * bnh_range)

    # Format right y-axis as currency
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

    # Add title
    title = f'{args.symbol} - Strategy Comparison (Dual Y-Axis)\n{args.start_date} to {args.end_date}'
    ax1.set_title(title, fontsize=14, fontweight='bold', pad=20)

    # Combine legends from both axes
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left', fontsize=10)

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')

    # Tight layout to prevent label cutoff
    plt.tight_layout()

    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def display_comparison_table(dma_bt, bnh_bt, args):
    """
    Display side-by-side comparison of strategies.

    Args:
        dma_bt: DMA backtester instance
        bnh_bt: Buy & Hold backtester instance
        args: Command-line arguments
    """
    dma_perf = dma_bt.slipped_performance
    bnh_perf = bnh_bt.slipped_performance

    # Define metrics to display
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
        ('Days Held (Wins)', 'days_held_wins', '.1f'),
        ('#Losses', 'num_losses', ',.0f'),
        ('Average PnL/Loss', 'avg_pnl_loss', ',.2f'),
        ('Days Held (Loss)', 'days_held_losses', '.1f'),
        ('Max Drawdown', 'max_drawdown', ',.2f'),
        ('Drawdown #2', 'drawdown_2', ',.2f'),
        ('Drawdown #3', 'drawdown_3', ',.2f'),
    ]

    # Calculate column widths
    metric_width = max(len(m[0]) for m in metrics) + 2
    value_width = 20

    # Print header
    print("=" * 80)
    print(f"STRATEGY COMPARISON: {args.symbol}")
    print(f"Period: {args.start_date} to {args.end_date}")
    print(f"Dollar Size: ${args.dollar_size:,.2f} | Slippage: {args.slippage} bps")
    print("=" * 80)
    print()

    # Print table header
    header = f"{'Metric':<{metric_width}} {f'{args.lookback}-DMA':>{value_width}} {'Buy & Hold':>{value_width}}"
    print(header)
    print("-" * len(header))

    # Print each metric
    for label, key, fmt in metrics:
        dma_val = dma_perf.get(key, 0)
        bnh_val = bnh_perf.get(key, 0)

        # Format values
        if 'f' in fmt:
            if key in ['total_pnl', 'mean_pnl_per_trade', 'avg_pnl_win', 'avg_pnl_loss',
                       'max_drawdown', 'drawdown_2', 'drawdown_3', 'pnl_per_exposure_day']:
                dma_str = f"${dma_val:{fmt}}"
                bnh_str = f"${bnh_val:{fmt}}"
            else:
                dma_str = f"{dma_val:{fmt}}"
                bnh_str = f"{bnh_val:{fmt}}"
        else:
            dma_str = str(dma_val)
            bnh_str = str(bnh_val)

        print(f"{label:<{metric_width}} {dma_str:>{value_width}} {bnh_str:>{value_width}}")

    print("=" * 80)

    # Print performance comparison
    print()
    print("PERFORMANCE DIFFERENCE (DMA - Buy & Hold):")
    print("-" * 80)

    pnl_diff = dma_perf['total_pnl'] - bnh_perf['total_pnl']
    pnl_diff_pct = (pnl_diff / abs(bnh_perf['total_pnl']) * 100) if bnh_perf['total_pnl'] != 0 else 0

    sharpe_diff = dma_perf['sharpe'] - bnh_perf['sharpe']

    print(f"Total PnL Difference:   ${pnl_diff:+,.2f} ({pnl_diff_pct:+.2f}%)")
    print(f"Sharpe Difference:      {sharpe_diff:+.3f}")
    print()

    # Interpretation
    if pnl_diff > 0:
        print(f"✓ DMA outperformed Buy & Hold by ${pnl_diff:,.2f}")
    elif pnl_diff < 0:
        print(f"✗ DMA underperformed Buy & Hold by ${-pnl_diff:,.2f}")
    else:
        print("○ DMA and Buy & Hold had identical returns")

    if sharpe_diff > 0:
        print(f"✓ DMA has better risk-adjusted returns (Sharpe: {sharpe_diff:+.3f})")
    elif sharpe_diff < 0:
        print(f"✗ Buy & Hold has better risk-adjusted returns (Sharpe: {sharpe_diff:+.3f})")
    else:
        print("○ DMA and Buy & Hold have identical Sharpe ratios")

    print()
    print("Note: Both strategies use identical volatility normalization methodology.")
    print("      The difference shows the value-add of DMA timing vs. being always long.")

    print("=" * 80)


def main():
    """Main function to run strategy comparison."""
    args = parse_arguments()

    # Configure logging
    log_level = logging.INFO if args.verbose else logging.WARNING
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Print header
    print()
    print("=" * 80)
    print("QUANTSTRAT STRATEGY COMPARISON")
    print("=" * 80)
    print(f"Symbol:         {args.symbol}")
    print(f"Strategies:     {args.lookback}-DMA vs Vol-Normalized Buy & Hold")
    print(f"Period:         {args.start_date} to {args.end_date}")
    print(f"Dollar Size:    ${args.dollar_size:,.2f}")
    print(f"Slippage:       {args.slippage} bps")
    print("=" * 80)
    print()

    try:
        # Run DMA backtest
        print(f"Running {args.lookback}-DMA backtest...")
        dma_strategy = DMA(lookback=args.lookback)
        dma_backtester = Backtester(dma_strategy, dollar_size=args.dollar_size)
        dma_backtester(args.symbol, args.start_date, args.end_date, slippage_bps=args.slippage)

        if dma_backtester.slipped_performance is None:
            print("ERROR: DMA backtest failed")
            sys.exit(1)

        print("✓ DMA backtest complete")

        # Run Buy & Hold backtest
        print("Running Vol-Normalized Buy & Hold backtest...")
        bnh_strategy = VolNormalizedBuyAndHold()
        bnh_backtester = Backtester(bnh_strategy, dollar_size=args.dollar_size)
        bnh_backtester(args.symbol, args.start_date, args.end_date, slippage_bps=args.slippage)

        if bnh_backtester.slipped_performance is None:
            print("ERROR: Buy & Hold backtest failed")
            sys.exit(1)

        print("✓ Buy & Hold backtest complete")
        print()

        # Display comparison
        display_comparison_table(dma_backtester, bnh_backtester, args)
        print()

        # Create visualizations
        comparison_output = f"output/{args.symbol}-comparison-dual-axis.png"

        print(f"Saving visualization...")
        create_dual_axis_plot(dma_backtester, bnh_backtester, args, comparison_output)
        print(f"  ✓ {comparison_output}")
        print()

    except KeyboardInterrupt:
        print()
        print("=" * 80)
        print("Comparison interrupted by user")
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
