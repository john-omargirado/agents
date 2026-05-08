import React, { useState, useMemo, useEffect } from 'react';
import { ChevronLeft, ChevronRight, BarChart3, RefreshCw, AlertCircle } from 'lucide-react';
import CandlestickChart from './CandlestickChart';
import BacktestingMetrics from './BacktestingMetrics';
import { getBacktestDates, runBacktestAnalysis, simulateTrade } from '../services/api';

const BACKTEST_PAIRS = [
    { value: 'EUR/USD', label: 'EUR/USD' },
    { value: 'USD/JPY', label: 'USD/JPY' },
    { value: 'GBP/USD', label: 'GBP/USD' },
    { value: 'AUD/USD', label: 'AUD/USD' },
    { value: 'USD/CAD', label: 'USD/CAD' },
    { value: 'USD/CHF', label: 'USD/CHF' },
    { value: 'USD/PHP', label: 'USD/PHP' },
];

/**
 * Backtesting view: Historical strategy performance analysis
 * Displays real daily signals, entry/exit prices, P&L, and cumulative performance
 */
export default function Backtesting() {
    const [pair, setPair] = useState('EUR/USD');
    const [availableDates, setAvailableDates] = useState([]);
    const [currentDateIndex, setCurrentDateIndex] = useState(-1);
    
    const [loading, setLoading] = useState(false);
    const [analysisResult, setAnalysisResult] = useState(null);
    const [simResult, setSimResult] = useState(null);
    const [error, setError] = useState(null);

    // Track session stats
    const [sessionResults, setSessionResults] = useState({}); // { [date]: dayData }

    // Fetch dates when pair changes
    useEffect(() => {
        async function loadDates() {
            setLoading(true);
            setAnalysisResult(null);
            setSimResult(null);
            setError(null);
            setSessionResults({});
            
            try {
                const data = await getBacktestDates(pair);
                if (data && data.dates && data.dates.length > 0) {
                    setAvailableDates(data.dates);
                    // Start from the most recent available backtest date
                    setCurrentDateIndex(data.dates.length - 1);
                } else {
                    setAvailableDates([]);
                    setCurrentDateIndex(-1);
                    setError("No backtest data available for this pair.");
                }
            } catch (err) {
                console.error("Failed to load backtest dates", err);
                setError("Failed to connect to backend. Make sure the server is running.");
            } finally {
                setLoading(false);
            }
        }
        loadDates();
    }, [pair]);

    // Run analysis when date changes
    useEffect(() => {
        if (currentDateIndex >= 0 && availableDates[currentDateIndex]) {
            runAnalysis(availableDates[currentDateIndex]);
        }
    }, [currentDateIndex, availableDates]);

    async function runAnalysis(date) {
        setLoading(true);
        setError(null);
        setAnalysisResult(null);
        setSimResult(null);

        try {
            // Use skip_llm=true for faster backtesting unless specific analysis is needed
            const result = await runBacktestAnalysis(pair, date, true);
            setAnalysisResult(result);

            let dayData = {
                date: new Date(date),
                signal: result.verdict.decision,
                entry: result.tts.price || 0,
                exit: result.tts.price || 0,
                pips: 0,
                pnl: 0,
                win: false,
                tradeTaken: result.verdict.decision !== 'HOLD'
            };

            if (result.verdict.decision !== 'HOLD' && result.tts.price && result.trade.sl_distance) {
                try {
                    const sim = await simulateTrade({
                        currencyPair: pair,
                        action: result.verdict.decision,
                        entryPrice: result.tts.price,
                        slDistance: result.trade.sl_distance,
                        tpDistance: result.trade.tp_distance,
                        targetDate: date
                    });
                    setSimResult(sim);
                    
                    dayData.exit = sim.exit_price;
                    dayData.pips = sim.pnl_pips;
                    dayData.win = sim.pnl_pips > 0;
                    
                    const pipsPerPoint = pair.includes('JPY') ? 100 : 10000;
                    dayData.pnl = (sim.pnl_pips / pipsPerPoint) * 1000; // Mock $1000 basis
                } catch (simErr) {
                    console.error("Simulation failed", simErr);
                }
            }
            
            setSessionResults(prev => ({
                ...prev,
                [date]: dayData
            }));

        } catch (err) {
            console.error("Backtest analysis failed", err);
            setError("Failed to analyze data for " + date);
        } finally {
            setLoading(false);
        }
    }

    // Navigation handlers
    const goToPreviousDay = () => {
        setCurrentDateIndex((prev) => Math.max(0, prev - 1));
    };

    const goToNextDay = () => {
        setCurrentDateIndex((prev) => Math.min(availableDates.length - 1, prev + 1));
    };

    const handlePairChange = (e) => {
        setPair(e.target.value);
    };

    const dayData = useMemo(() => {
        if (!analysisResult) return null;
        return sessionResults[availableDates[currentDateIndex]];
    }, [analysisResult, sessionResults, currentDateIndex, availableDates]);

    const stats = useMemo(() => {
        const results = Object.values(sessionResults);
        let totalPnl = 0;
        let wins = 0;
        let losses = 0;
        let holds = 0;
        let totalTrades = 0;

        results.forEach(r => {
            totalPnl += r.pnl;
            if (r.tradeTaken) {
                totalTrades++;
                if (r.win) wins++;
                else losses++;
            } else {
                holds++;
            }
        });

        const winRate = totalTrades > 0 ? ((wins / totalTrades) * 100).toFixed(1) : 0;

        return {
            totalPnl: Number(totalPnl.toFixed(2)),
            wins,
            losses,
            holds,
            totalTrades,
            winRate: parseFloat(winRate),
        };
    }, [sessionResults]);

    const displayDate = availableDates[currentDateIndex] || '';

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
                                disabled={currentDateIndex <= 0 || loading}
                                aria-label="Previous day"
                            >
                                <ChevronLeft size={18} />
                            </button>

                            <input
                                id="date-picker"
                                type="date"
                                value={displayDate}
                                readOnly
                                className="date-input"
                            />

                            <button
                                className="nav-btn"
                                onClick={goToNextDay}
                                disabled={currentDateIndex >= availableDates.length - 1 || loading}
                                aria-label="Next day"
                            >
                                <ChevronRight size={18} />
                            </button>

                            <span className="day-counter">
                                {availableDates.length > 0 
                                    ? `Day ${currentDateIndex + 1} / ${availableDates.length}`
                                    : 'No data'}
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
                            {pair} — {displayDate || '...'}
                        </h2>
                        <p>Backtest Candlestick View</p>
                    </div>
                    
                    <CandlestickChart
                        pair={pair}
                        date={displayDate}
                        theme="dark"
                    />

                    {loading && (
                        <div className="chart-overlay-loading">
                            <RefreshCw className="spinner" size={32} />
                            <p>Analyzing historical data...</p>
                        </div>
                    )}
                    
                    {error && (
                        <div className="chart-overlay-error">
                            <AlertCircle size={32} />
                            <p>{error}</p>
                            <button onClick={() => runAnalysis(displayDate)} className="btn-retry">
                                <RefreshCw size={14} /> Retry
                            </button>
                        </div>
                    )}
                </div>

                {/* RIGHT: Metrics Sidebar */}
                <BacktestingMetrics dayData={dayData} stats={stats} />
            </div>
        </div>
    );
}

