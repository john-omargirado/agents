import React, { useState, useMemo } from 'react';
import { ChevronLeft, ChevronRight, BarChart3 } from 'lucide-react';
import CandlestickChart from './CandlestickChart';
import BacktestingMetrics from './BacktestingMetrics';
import { generateBacktestDay, generateBacktestStats, BACKTEST_PAIRS } from '../utils/backtestingData';

const TOTAL_BACKTEST_DAYS = 64;

/**
 * Backtesting view: Historical strategy performance analysis
 * Displays daily signals, entry/exit prices, P&L, and cumulative performance
 * Navigable by day (1-64) with pair selector
 */
export default function Backtesting() {
    const [pair, setPair] = useState('EUR/USD');
    const [currentDay, setCurrentDay] = useState(1);

    // Generate data for current day and cumulative stats
    const dayData = useMemo(
        () => generateBacktestDay(pair, currentDay, TOTAL_BACKTEST_DAYS),
        [pair, currentDay]
    );

    const stats = useMemo(
        () => generateBacktestStats(pair, TOTAL_BACKTEST_DAYS),
        [pair]
    );

    // Navigation handlers
    const goToPreviousDay = () => {
        setCurrentDay((prev) => Math.max(1, prev - 1));
    };

    const goToNextDay = () => {
        setCurrentDay((prev) => Math.min(TOTAL_BACKTEST_DAYS, prev + 1));
    };

    const handlePairChange = (e) => {
        setPair(e.target.value);
        setCurrentDay(1); // Reset to first day when switching pairs
    };

    // Format date for display
    const displayDate = dayData.date
        .toLocaleDateString('en-CA', {
            year: 'numeric',
            month: '2-digit',
            day: '2-digit',
        });

    return (
        <div className="backtesting-container">
            {/* HEADER */}
            <div className="backtesting-header">
                <div className="header-left">
                    <BarChart3 size={24} className="header-icon" />
                    <div>
                        <h1>Backtesting</h1>
                        <p>Historical Strategy Performance</p>
                    </div>
                </div>

                <div className="header-controls pair-selector">
                    <div className="control-group">
                        <label htmlFor="pair-select">Pair:</label>
                        <select
                            id="pair-select"
                            className="form-select"
                            value={pair}
                            onChange={handlePairChange}
                        >
                            {BACKTEST_PAIRS.map((p) => (
                                <option key={p.value} value={p.value}>
                                    {p.label}
                                </option>
                            ))}
                        </select>
                    </div>
                </div>

                <div className="header-controls date-selector">
                    <div className="control-group">
                        <label htmlFor="date-picker">Date:</label>
                        <div className="date-nav">
                            <button
                                className="nav-btn"
                                onClick={goToPreviousDay}
                                disabled={currentDay === 1}
                                aria-label="Previous day"
                            >
                                <ChevronLeft size={18} />
                            </button>

                            <input
                                id="date-picker"
                                type="date"
                                value={displayDate}
                                onChange={(e) => {
                                    // Allow manual date input (simplified)
                                    const selected = new Date(e.target.value);
                                    const baseDate = new Date(2024, 11, 31);
                                    const dayDiff = Math.floor(
                                        (baseDate - selected) / (1000 * 60 * 60 * 24)
                                    );
                                    const newDay = TOTAL_BACKTEST_DAYS - dayDiff;
                                    if (newDay >= 1 && newDay <= TOTAL_BACKTEST_DAYS) {
                                        setCurrentDay(newDay);
                                    }
                                }}
                                disabled
                                className="date-input"
                            />

                            <button
                                className="nav-btn"
                                onClick={goToNextDay}
                                disabled={currentDay === TOTAL_BACKTEST_DAYS}
                                aria-label="Next day"
                            >
                                <ChevronRight size={18} />
                            </button>

                            <span className="day-counter">
                                Day {currentDay} / {TOTAL_BACKTEST_DAYS}
                            </span>
                        </div>
                    </div>
                </div>
            </div>

            {/* MAIN CONTENT */}
            <div className="backtesting-content">
                {/* LEFT: Chart */}
                <div className="backtesting-chart-area">
                    <div className="chart-header">
                        <h2>
                            {pair} — {displayDate}
                        </h2>
                        <p>Backtest Candlestick View</p>
                    </div>
                    <CandlestickChart
                        pair={pair}
                        ohlcvData={dayData.ohlcv}
                        theme="dark"
                    />
                </div>

                {/* RIGHT: Metrics Sidebar */}
                <BacktestingMetrics dayData={dayData} stats={stats} />
            </div>
        </div>
    );
}
