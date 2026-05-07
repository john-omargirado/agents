import React, { useEffect, useMemo, useRef, useState } from 'react';
import { Calendar, ChevronLeft, ChevronRight, X, BarChart3 } from 'lucide-react';
import { createChart } from 'lightweight-charts';

const MIN_DATE = '2022-01-01';
const MAX_DATE = '2025-12-31';

function createSeedFromText(text) {
    let seed = 0;
    for (let i = 0; i < text.length; i += 1) {
        seed = (seed * 31 + text.charCodeAt(i)) >>> 0;
    }
    return seed || 123456789;
}

function mulberry32(seed) {
    let t = seed >>> 0;
    return () => {
        t += 0x6d2b79f5;
        let r = Math.imul(t ^ (t >>> 15), t | 1);
        r ^= r + Math.imul(r ^ (r >>> 7), r | 61);
        return ((r ^ (r >>> 14)) >>> 0) / 4294967296;
    };
}

// FIX 1: days bumped from 450 → 1461 to cover the full 2022-2025 range
// (3 regular years + 1 leap year = 1461 days)
function generateSyntheticCandles(pair, days = 1461, basePrice = 1.085) {
    const candles = [];
    const random = mulberry32(createSeedFromText(pair || 'EUR/USD'));
    // Anchor synthetic candles starting from MIN_DATE
    const start = new Date(MIN_DATE);
    start.setUTCHours(0, 0, 0, 0);

    let price = basePrice + (random() - 0.5) * 0.03;

    for (let i = 0; i < days; i += 1) {
        const d = new Date(start);
        d.setUTCDate(start.getUTCDate() + i);
        const time = Math.floor(d.getTime() / 1000);

        const open = price;
        const volatility = 0.001 + random() * 0.002;
        const direction = random() > 0.49 ? 1 : -1;
        const close = open + direction * volatility;
        const high = Math.max(open, close) + random() * 0.0012;
        const low = Math.min(open, close) - random() * 0.0012;

        candles.push({
            time,
            open: Number(open.toFixed(4)),
            high: Number(high.toFixed(4)),
            low: Number(low.toFixed(4)),
            close: Number(close.toFixed(4)),
        });

        price = close;
    }

    return candles;
}

function normalizeCandleData(rawRows) {
    if (!Array.isArray(rawRows)) return [];

    return rawRows
        .map((row) => {
            const timeValue = row.time ?? row.timestamp ?? row.date;
            let time;
            if (typeof timeValue === 'number') {
                time = timeValue > 9999999999 ? Math.floor(timeValue / 1000) : timeValue;
            } else if (typeof timeValue === 'string') {
                const parsed = new Date(timeValue).getTime();
                if (!Number.isFinite(parsed)) return null;
                time = Math.floor(parsed / 1000);
            } else {
                return null;
            }

            const open = Number(row.open);
            const high = Number(row.high);
            const low = Number(row.low);
            const close = Number(row.close);
            if (![open, high, low, close].every(Number.isFinite)) return null;

            return { time, open, high, low, close };
        })
        .filter(Boolean)
        .sort((a, b) => a.time - b.time);
}

function toPairSymbol(pair) {
    return (pair || '').replace('/', '').toUpperCase();
}

/**
 * Convert "YYYY-MM-DD" → Unix seconds at end of that UTC day (23:59:59).
 */
function dateStrToEndOfDayTs(dateStr) {
    const [y, m, d] = dateStr.split('-').map(Number);
    return Math.floor(Date.UTC(y, m - 1, d + 1) / 1000) - 1;
}

/**
 * Convert "YYYY-MM-DD" → Unix seconds at the START of that UTC day (00:00:00).
 */
function dateStrToStartOfDayTs(dateStr) {
    const [y, m, d] = dateStr.split('-').map(Number);
    return Math.floor(Date.UTC(y, m - 1, d) / 1000);
}

function formatDisplayDate(dateStr) {
    if (!dateStr) return null;
    const [y, m, d] = dateStr.split('-').map(Number);
    const date = new Date(Date.UTC(y, m - 1, d));
    return date.toLocaleDateString('en-US', {
        year: 'numeric', month: 'short', day: 'numeric',
        timeZone: 'UTC',
    });
}

// ── Component ────────────────────────────────────────────────────────────────

export default function CandlestickChart({ pair, ohlcvData, theme = 'dark', onDateChange }) {

    // ── Date picker — starts at Jan 1 2022 ──────────────────────────────────
    const [selectedDate, setSelectedDate] = useState(MIN_DATE);

    const handleDateChange = (e) => {
        const val = e.target.value;
        setSelectedDate(val);
        if (onDateChange) onDateChange(val || null);
    };

    const clearDate = () => {
        setSelectedDate('');
        if (onDateChange) onDateChange(null);
    };

    const stepDate = (direction) => {
        const base = selectedDate || MIN_DATE;
        const [y, m, d] = base.split('-').map(Number);
        const date = new Date(Date.UTC(y, m - 1, d));
        date.setUTCDate(date.getUTCDate() + direction);
        const newDate = date.toISOString().split('T')[0];
        if (newDate < MIN_DATE || newDate > MAX_DATE) return;
        setSelectedDate(newDate);
        if (onDateChange) onDateChange(newDate);
    };

    // ── Load real candle data from JSON file ─────────────────────────────────
    const [jsonCandles, setJsonCandles] = useState(null);

    useEffect(() => {
        if (!pair) return;
        const symbol = toPairSymbol(pair);
        setJsonCandles(null);

        fetch(`/data/backtesting/forex_pairs/${symbol}.json`)
            .then((res) => {
                if (!res.ok) throw new Error(`HTTP ${res.status}`);
                return res.json();
            })
            .then((raw) => {
                const rows = Array.isArray(raw) ? raw : (raw.data ?? raw.candles ?? raw.ohlcv ?? []);
                const normalized = normalizeCandleData(rows);
                setJsonCandles(normalized.length > 0 ? normalized : null);
            })
            .catch(() => setJsonCandles(null));
    }, [pair]);

    // ── Chart DOM refs ───────────────────────────────────────────────────────
    const chartContainerRef = useRef(null);
    const chartRef = useRef(null);
    const candleSeriesRef = useRef(null);

    // ── Accessible palette toggle ────────────────────────────────────────────
    const [useAccessiblePalette, setUseAccessiblePalette] = useState(false);

    useEffect(() => {
        const saved = window.localStorage.getItem('trading_assistant_candle_palette');
        setUseAccessiblePalette(saved === 'accessible');
    }, []);

    const toggleAccessiblePalette = () => {
        setUseAccessiblePalette((prev) => {
            const next = !prev;
            window.localStorage.setItem('trading_assistant_candle_palette', next ? 'accessible' : 'default');
            return next;
        });
    };

    // ── Full candle dataset (priority: JSON file > prop > synthetic) ─────────
    const allCandleData = useMemo(() => {
        if (jsonCandles && jsonCandles.length > 0) return jsonCandles;
        const normalized = normalizeCandleData(ohlcvData);
        if (normalized.length > 0) return normalized;
        return generateSyntheticCandles(pair);
    }, [jsonCandles, ohlcvData, pair]);

    // ── Price stats — derived from the last candle on/before selected date ───
    const visibleCandleData = useMemo(() => {
        if (!allCandleData?.length) return [];

        if (!selectedDate) return allCandleData;

        const cutoff = dateStrToEndOfDayTs(selectedDate);

        const sliced = allCandleData.filter((c) => c.time <= cutoff);

        // FIX: if cutoff is invalid or too early, DO NOT fallback to synthetic logic
        // instead clamp to first available candle before crashing logic
        if (sliced.length === 0) return [allCandleData[0]];

        return sliced;
    }, [allCandleData, selectedDate]);

    const currentCandle = visibleCandleData[visibleCandleData.length - 1] || null;
    const previousCandle = visibleCandleData[visibleCandleData.length - 2] || currentCandle;
    const currentPrice = currentCandle?.close ?? null;
    const priceChange = (currentCandle && previousCandle) ? currentCandle.close - previousCandle.close : 0;
    const priceChangePct = previousCandle?.close ? (priceChange / previousCandle.close) * 100 : 0;
    const isPositive = priceChange >= 0;

    // ── Chart theme ──────────────────────────────────────────────────────────
    const chartPalette = useMemo(() => {
        if (theme === 'light') {
            return {
                textColor: '#475569',
                gridColor: 'rgba(148, 163, 184, 0.28)',
                borderColor: '#cbd5e1',
                crosshairColor: 'rgba(14, 106, 99, 0.35)',
                crosshairLabelBg: '#0e6a63',
            };
        }
        return {
            textColor: '#64748b',
            gridColor: 'rgba(30, 41, 59, 0.5)',
            borderColor: '#1e293b',
            crosshairColor: 'rgba(45, 212, 191, 0.3)',
            crosshairLabelBg: '#2dd4bf',
        };
    }, [theme]);

    const candlePalette = useMemo(() => {
        if (useAccessiblePalette) {
            return {
                upColor: '#2563eb', downColor: '#ea580c',
                borderUpColor: '#1d4ed8', borderDownColor: '#c2410c',
                wickUpColor: '#1e40af', wickDownColor: '#9a3412',
            };
        }
        return {
            upColor: '#22c55e', downColor: '#ef4444',
            borderUpColor: '#22c55e', borderDownColor: '#ef4444',
            wickUpColor: '#22c55e', wickDownColor: '#ef4444',
        };
    }, [useAccessiblePalette]);

    // ── Effect 1: Build/rebuild chart when data or palette changes ───────────
    // Does NOT depend on selectedDate — date scrolling is handled by Effect 2.
    useEffect(() => {
        if (!chartContainerRef.current) return;

        if (chartRef.current) {
            chartRef.current.remove();
            chartRef.current = null;
            candleSeriesRef.current = null;
        }

        const chart = createChart(chartContainerRef.current, {
            width: chartContainerRef.current.clientWidth,
            height: chartContainerRef.current.clientHeight,
            layout: {
                background: { color: 'transparent' },
                textColor: chartPalette.textColor,
                fontFamily: "'Inter', sans-serif",
                fontSize: 11,
            },
            grid: {
                vertLines: { color: chartPalette.gridColor },
                horzLines: { color: chartPalette.gridColor },
            },
            crosshair: {
                mode: 0,
                vertLine: { color: chartPalette.crosshairColor, labelBackgroundColor: chartPalette.crosshairLabelBg },
                horzLine: { color: chartPalette.crosshairColor, labelBackgroundColor: chartPalette.crosshairLabelBg },
            },
            rightPriceScale: {
                borderColor: chartPalette.borderColor,
                scaleMargins: { top: 0.1, bottom: 0.1 },
            },
            timeScale: {
                borderColor: chartPalette.borderColor,
                timeVisible: true,
                secondsVisible: false,
                rightOffset: 2,
            },
            handleScroll: false,
            handleScale: false,
        });

        // Load ALL candles — the visible window is positioned by Effect 2
        const candleSeries = chart.addCandlestickSeries(candlePalette);
        candleSeries.setData(allCandleData);

        chartRef.current = chart;
        candleSeriesRef.current = candleSeries;

        const handleResize = () => {
            if (chartRef.current && chartContainerRef.current) {
                chartRef.current.applyOptions({
                    width: chartContainerRef.current.clientWidth,
                    height: chartContainerRef.current.clientHeight,
                });
            }
        };

        const resizeObserver = new ResizeObserver(handleResize);
        resizeObserver.observe(chartContainerRef.current);

        return () => {
            resizeObserver.disconnect();
            if (chartRef.current) {
                chartRef.current.remove();
                chartRef.current = null;
            }
            candleSeriesRef.current = null;
        };
    }, [allCandleData, chartPalette, candlePalette]);

    // ── Effect 2: Pan the time scale to the selected date ────────────────────
    // FIX 2: explicit deps listed — React will never skip this when date changes
    useEffect(() => {
        if (!chartRef.current || !allCandleData.length) return;

        const cutoff = selectedDate
            ? dateStrToEndOfDayTs(selectedDate)
            : allCandleData[allCandleData.length - 1].time;

        const filtered = allCandleData.filter((c) => c.time <= cutoff);

        if (!filtered.length) {
            chartRef.current.timeScale().fitContent();
            return;
        }

        const WINDOW = 60;
        const startIdx = Math.max(0, filtered.length - WINDOW);

        const from = filtered[startIdx].time;
        const to = filtered[filtered.length - 1].time;

        chartRef.current.timeScale().setVisibleRange({
            from,
            to: to + 43200,
        });

    }, [selectedDate, allCandleData]); // explicit deps — no eslint-disable needed

    // ── Data source label ────────────────────────────────────────────────────
    const dataSourceLabel = useMemo(() => {
        if (jsonCandles && jsonCandles.length > 0) return `${toPairSymbol(pair)} · Live Data`;
        if (normalizeCandleData(ohlcvData).length > 0) return 'Provided Data';
        return 'Demo Market Candles';
    }, [jsonCandles, ohlcvData, pair]);

    // ── Render ───────────────────────────────────────────────────────────────
    return (
        <div className="card chart-card">

            {/* ── Chart header ── */}
            <div className="chart-header">
                <div className="chart-pair">
                    <div className="chart-pair-icon">
                        <BarChart3 size={18} />
                    </div>
                    <div>
                        <h2>{pair}</h2>
                        <span className="chart-timeframe">{dataSourceLabel}</span>
                    </div>
                </div>
                <div className="chart-right-controls">
                    <div className="chart-legend" aria-label="Candle direction legend">
                        <span className={`chart-legend-item ${useAccessiblePalette ? 'accessible-up' : 'default-up'}`}>
                            <span className="legend-swatch" />
                            <span>▲ Up</span>
                        </span>
                        <span className={`chart-legend-item ${useAccessiblePalette ? 'accessible-down' : 'default-down'}`}>
                            <span className="legend-swatch" />
                            <span>▼ Down</span>
                        </span>
                    </div>
                    <div className="chart-price-info">
                        <button
                            className="chart-a11y-btn"
                            type="button"
                            onClick={toggleAccessiblePalette}
                            aria-pressed={useAccessiblePalette}
                        >
                            {useAccessiblePalette ? 'Use Standard Candles' : 'Use Color-Safe Candles'}
                        </button>
                        <div className="chart-price">{currentPrice === null ? '--' : currentPrice.toFixed(4)}</div>
                        {currentPrice === null ? (
                            <div className="chart-change">No market data</div>
                        ) : (
                            <div className={`chart-change ${isPositive ? 'positive' : 'negative'}`}>
                                {isPositive ? '▲ ' : '▼ '}{isPositive ? '+' : ''}{priceChange.toFixed(4)} ({isPositive ? '+' : ''}{priceChangePct.toFixed(2)}%)
                            </div>
                        )}
                    </div>
                </div>
            </div>

            {/* ── Candlestick chart ── */}
            <div className="chart-container" ref={chartContainerRef} />

            {/* ── Date picker section ── */}
            <div className="date-picker-section">
                <div className="date-picker-label">
                    Analysis Date
                    <span className="date-picker-range-hint">Jan 2022 – Dec 2025</span>
                </div>

                <div className="date-picker-controls">
                    <button
                        className="date-step-btn"
                        onClick={() => stepDate(-1)}
                        title="Previous day"
                        disabled={selectedDate === MIN_DATE}
                    >
                        <ChevronLeft size={16} />
                    </button>

                    <div className="date-input-wrap">
                        <Calendar size={14} className="date-input-icon" />
                        <input
                            type="date"
                            className="date-input"
                            value={selectedDate}
                            min={MIN_DATE}
                            max={MAX_DATE}
                            onChange={handleDateChange}
                        />
                        {selectedDate && (
                            <button className="date-clear-btn" onClick={clearDate} title="Clear date">
                                <X size={12} />
                            </button>
                        )}
                    </div>

                    <button
                        className="date-step-btn"
                        onClick={() => stepDate(1)}
                        title="Next day"
                        disabled={selectedDate === MAX_DATE}
                    >
                        <ChevronRight size={16} />
                    </button>
                </div>

                <div className="date-picker-display">
                    {selectedDate ? (
                        <span className="date-selected-label">
                            <span className="date-dot active" />
                            Analyzing: <strong>{formatDisplayDate(selectedDate)}</strong>
                        </span>
                    ) : (
                        <span className="date-selected-label muted">
                            <span className="date-dot" />
                            No date selected — showing all available data
                        </span>
                    )}
                </div>

                {/* FIX 3: Quick shortcuts — 2025 now points to Jan 2 2025 (safe for all data sources) */}
                <div className="date-shortcuts">
                    <span className="date-shortcuts-label">Quick select:</span>
                    {[
                        { label: '2022', date: '2022-01-01' },
                        { label: '2023', date: '2023-01-02' },
                        { label: '2024', date: '2024-01-02' },
                        { label: '2025', date: '2025-01-02' },
                    ].map(({ label, date }) => (
                        <button
                            key={label}
                            className={`date-shortcut-btn ${selectedDate === date ? 'active' : ''}`}
                            onClick={() => {
                                setSelectedDate(date);
                                if (onDateChange) onDateChange(date);
                            }}
                        >
                            {label}
                        </button>
                    ))}
                </div>

                {/* Range bar */}
                <div className="date-range-bar">
                    <span className="date-range-start">Jan 1, 2022</span>
                    <div className="date-range-track">
                        <div
                            className="date-range-fill"
                            style={{
                                width: selectedDate
                                    ? `${Math.max(2, Math.min(100, ((new Date(selectedDate) - new Date(MIN_DATE)) / (new Date(MAX_DATE) - new Date(MIN_DATE))) * 100))}%`
                                    : '0%',
                            }}
                        />
                    </div>
                    <span className="date-range-end">Dec 31, 2025</span>
                </div>
            </div>

        </div>
    );
}