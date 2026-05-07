import React from 'react';
import DaySignalBadge from './DaySignalBadge';
import MetricsRow from './MetricsRow';
import MetricsSection from './MetricsSection';

/**
 * Right sidebar showing current day's trade metrics and cumulative performance
 * Displays entry/exit prices, P&L, and win rate statistics
 */
export default function BacktestingMetrics({ dayData, stats }) {
    if (!dayData || !stats) {
        return <div className="backtest-metrics loading">Loading metrics...</div>;
    }

    const { signal, entry, exit, pips, pnl, win } = dayData;
    const { totalPnl, winRate, wins, losses, holds, totalTrades } = stats;

    // Determine if P&L is positive or negative
    const pnlClass = pnl > 0 ? 'positive' : pnl < 0 ? 'negative' : 'neutral';

    return (
        <div className="backtest-metrics">
            {/* DAY SIGNAL */}
            <MetricsSection title="DAY SIGNAL">
                <div className="metrics-content">
                    <DaySignalBadge signal={signal} />
                </div>
            </MetricsSection>

            {/* ENTRY / EXIT */}
            <MetricsSection title="ENTRY / EXIT">
                <MetricsRow label="Entry" value={entry.toFixed(5)} />
                <MetricsRow label="Exit" value={exit.toFixed(5)} />
            </MetricsSection>

            {/* PIPS & P&L */}
            <MetricsSection title="PIPS & P&L">
                <MetricsRow
                    label="Pips"
                    value={`${pips > 0 ? '+' : ''}${pips}`}
                    valueClass={`pips-${pips > 0 ? 'positive' : 'negative'}`}
                />
                <MetricsRow
                    label="P&L"
                    value={`${pnl > 0 ? '+' : ''}${pnl.toFixed(2)}`}
                    valueClass={`pnl-${pnlClass}`}
                />
            </MetricsSection>

            {/* CUMULATIVE PERFORMANCE */}
            <MetricsSection title="CUMULATIVE PERFORMANCE" className="cumulative">
                <MetricsRow
                    label="Total P&L"
                    value={`${totalPnl > 0 ? '+' : ''}${totalPnl.toFixed(2)}`}
                    valueClass={`pnl-${totalPnl > 0 ? 'positive' : 'negative'}`}
                />
                <MetricsRow
                    label="Win Rate"
                    value={`${winRate}%`}
                />
                <MetricsRow
                    label="Wins / Losses / Holds"
                    value={
                        <span>
                            <span className="win">{wins}</span>
                            {' / '}
                            <span className="loss">{losses}</span>
                            {' / '}
                            <span className="hold">{holds}</span>
                        </span>
                    }
                />
                <MetricsRow
                    label="Total Trades"
                    value={totalTrades}
                />
            </MetricsSection>
        </div>
    );
}
