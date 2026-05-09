import React, { useEffect, useMemo, useRef, useState } from 'react';
import { Calendar, ChevronLeft, ChevronRight, X, BarChart3 } from 'lucide-react';
import { createChart } from 'lightweight-charts';

const PAIR_CONFIG = {
    // Major USD pairs
    EURUSD: { basePrice: 1.085, decimals: 5 },
    GBPUSD: { basePrice: 1.265, decimals: 5 },
    AUDUSD: { basePrice: 0.645, decimals: 5 },
    NZDUSD: { basePrice: 0.595, decimals: 5 },
    USDCAD: { basePrice: 1.355, decimals: 5 },
    USDCHF: { basePrice: 0.895, decimals: 5 },
    // JPY pairs — 3 decimals, price ~100–200 range
    USDJPY: { basePrice: 149.5, decimals: 3 },
    EURJPY: { basePrice: 160.0, decimals: 3 },
    GBPJPY: { basePrice: 189.0, decimals: 3 },
    AUDJPY: { basePrice: 97.0, decimals: 3 },
    CADJPY: { basePrice: 110.5, decimals: 3 },
    CHFJPY: { basePrice: 167.0, decimals: 3 },
    NZDJPY: { basePrice: 91.0, decimals: 3 },
    // Cross pairs
    EURGBP: { basePrice: 0.857, decimals: 5 },
    EURAUD: { basePrice: 1.680, decimals: 5 },
    EURCAD: { basePrice: 1.475, decimals: 5 },
    GBPAUD: { basePrice: 1.960, decimals: 5 },
    GBPCAD: { basePrice: 1.720, decimals: 5 },
    AUDCAD: { basePrice: 0.875, decimals: 5 },
    // ── PHP Exotic pairs ────────────────────────────────────────────────────
    // USD/PHP: peso trades ~55–58 per dollar; 2 decimal places is standard
    USDPHP: { basePrice: 56.5, decimals: 2 },
    // EUR/PHP: real JSON data opens ~58.x (2022-01-03); used only as synthetic fallback
    EURPHP: { basePrice: 58.1, decimals: 2 },
    // JPY/PHP: inverted USDJPY × USDPHP — 1 JPY ≈ 0.378 PHP
    JPYPHP: { basePrice: 0.378, decimals: 4 },
};

// Category metadata — used for chart header badge and spread warning
const PAIR_META = {
    EURUSD: { category: 'Major' }, GBPUSD: { category: 'Major' },
    AUDUSD: { category: 'Major' }, NZDUSD: { category: 'Major' },
    USDCAD: { category: 'Major' }, USDCHF: { category: 'Major' },
    USDJPY: { category: 'Major' }, EURJPY: { category: 'Major' },
    GBPJPY: { category: 'Major' }, AUDJPY: { category: 'Major' },
    CADJPY: { category: 'Major' }, CHFJPY: { category: 'Major' },
    NZDJPY: { category: 'Major' }, EURGBP: { category: 'Cross' },
    EURAUD: { category: 'Cross' }, EURCAD: { category: 'Cross' },
    GBPAUD: { category: 'Cross' }, GBPCAD: { category: 'Cross' },
    AUDCAD: { category: 'Cross' },
    USDPHP: { category: 'Exotic' },
    EURPHP: { category: 'Exotic' },
    JPYPHP: { category: 'Exotic' },
};

function getPairConfig(pair) {
    const key = toPairSymbol(pair);
    return PAIR_CONFIG[key] ?? { basePrice: 1.085, decimals: 5 };
}

function getPairMeta(pair) {
    return PAIR_META[toPairSymbol(pair)] ?? { category: 'Major' };
}

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

function generateSyntheticCandles(pair, days = 1461) {
    const { basePrice, decimals } = getPairConfig(pair);
    const symbol = toPairSymbol(pair);
    const isJPY = symbol.includes('JPY') && !symbol.endsWith('PHP'); // USDJPY etc.
    const isPHP = symbol.endsWith('PHP');                             // *PHP exotics
    const isJPYPHP = symbol === 'JPYPHP';                               // tiny prices

    const candles = [];
    const random = mulberry32(createSeedFromText(pair || 'EUR/USD'));
    const start = new Date(MIN_DATE);
    start.setUTCHours(0, 0, 0, 0);

    let price = basePrice + (random() - 0.5) * (
        isJPYPHP ? 0.005 :
            isPHP ? 0.4 :
                isJPY ? 3.0 : 0.03
    );

    for (let i = 0; i < days; i += 1) {
        const d = new Date(start);
        d.setUTCDate(start.getUTCDate() + i);
        const time = Math.floor(d.getTime() / 1000);

        const open = price;

        // Volatility tuned per pair type
        let volBase, volRange, wickRange;
        if (isJPYPHP) {
            volBase = 0.0005; volRange = 0.001; wickRange = 0.0003;
        } else if (isPHP) {
            volBase = 0.04; volRange = 0.08; wickRange = 0.03;
        } else if (isJPY) {
            volBase = 0.15; volRange = 0.25; wickRange = 0.12;
        } else {
            volBase = 0.0008; volRange = 0.0015; wickRange = 0.0012;
        }

        const volatility = volBase + random() * volRange;
        const direction = random() > 0.49 ? 1 : -1;
        const close = open + direction * volatility;
        const high = Math.max(open, close) + random() * wickRange;
        const low = Math.min(open, close) - random() * wickRange;

        candles.push({
            time,
            open: Number(open.toFixed(decimals)),
            high: Number(high.toFixed(decimals)),
            low: Number(low.toFixed(decimals)),
            close: Number(close.toFixed(decimals)),
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

function dateStrToEndOfDayTs(dateStr) {
    const [y, m, d] = dateStr.split('-').map(Number);
    return Math.floor(Date.UTC(y, m - 1, d + 1) / 1000) - 1;
}

function isWeekend(dateStr) {
    const [y, m, d] = dateStr.split('-').map(Number);
    return [0, 6].includes(new Date(Date.UTC(y, m - 1, d)).getUTCDay());
}

function snapToWeekday(dateStr) {
    const [y, m, d] = dateStr.split('-').map(Number);
    const date = new Date(Date.UTC(y, m - 1, d));
    const day = date.getUTCDay();
    if (day === 6) date.setUTCDate(d - 1);
    if (day === 0) date.setUTCDate(d - 2);
    return date.toISOString().split('T')[0];
}

function formatDisplayDate(dateStr) {
    if (!dateStr) return null;
    const [y, m, d] = dateStr.split('-').map(Number);
    const date = new Date(Date.UTC(y, m - 1, d));
    return date.toLocaleDateString('en-US', {
        year: 'numeric', month: '2-digit', day: '2-digit',
        timeZone: 'UTC',
    });
}

// ── Component ────────────────────────────────────────────────────────────────

export default function CandlestickChart({ pair, ohlcvData, theme = 'dark', onDateChange, date }) {

    // ── Date picker — starts at Jan 1 2022 ──────────────────────────────────
    const [selectedDate, setSelectedDate] = useState(date || '2022-01-03');

    // Sync internal selectedDate when date prop changes
    useEffect(() => {
        if (date && date !== selectedDate) {
            setSelectedDate(date);
        }
    }, [date]);

    const handleDateChange = (e) => {
        const val = e.target.value;
        if (!val) { setSelectedDate(''); if (onDateChange) onDateChange(null); return; }
        const safe = isWeekend(val) ? snapToWeekday(val) : val;
        setSelectedDate(safe);
        if (onDateChange) onDateChange(safe);
    };

    const clearDate = () => {
        setSelectedDate('');
        if (onDateChange) onDateChange(null);
    };

    const stepDate = (direction) => {
        const base = selectedDate || '2022-01-03';
        const [y, m, d] = base.split('-').map(Number);
        const date = new Date(Date.UTC(y, m - 1, d));
        do {
            date.setUTCDate(date.getUTCDate() + direction);
        } while (isWeekend(date.toISOString().split('T')[0]));
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
    const dateInputRef = useRef(null);
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
            window.localStorage.setItem(
                'trading_assistant_candle_palette',
                next ? 'accessible' : 'default'
            );
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

    const pairDecimals = useMemo(() => getPairConfig(pair).decimals, [pair]);
    const pairMeta = useMemo(() => getPairMeta(pair), [pair]);
    const isExotic = pairMeta.category === 'Exotic';

    // ── Price stats ──────────────────────────────────────────────────────────
    const visibleCandleData = useMemo(() => {
        if (!allCandleData?.length) return [];
        if (!selectedDate) return allCandleData;
        const cutoff = dateStrToEndOfDayTs(selectedDate);
        const sliced = allCandleData.filter((c) => c.time <= cutoff);
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

    // ── Effect 1: Build/rebuild chart ────────────────────────────────────────
    useEffect(() => {
        if (!chartContainerRef.current) return;

        if (chartRef.current) {
            chartRef.current.remove();
            chartRef.current = null;
            candleSeriesRef.current = null;
        }
        const el = chartContainerRef.current;
        const chart = createChart(el, {
            width: el.clientWidth,
            height: Math.max(el.clientHeight, 160),
            autoSize: true,
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

        const candleSeries = chart.addCandlestickSeries(candlePalette);
        candleSeries.setData(allCandleData);

        chartRef.current = chart;
        candleSeriesRef.current = candleSeries;

        const handleResize = () => {
            if (!chartRef.current || !chartContainerRef.current) return;
            const { clientWidth, clientHeight } = chartContainerRef.current;
            chartRef.current.applyOptions({
                width: clientWidth,
                height: Math.max(clientHeight, 160), // floor prevents collapse to 0
            });
            chartRef.current.timeScale().fitContent();  // re-fit after resize
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

    // ── Effect 2: Pan to selected date ───────────────────────────────────────
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

        chartRef.current.timeScale().setVisibleRange({ from, to: to + 43200 });
    }, [selectedDate, allCandleData]);

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
                        <h2>
                            {pair}
                            {/* Exotic badge — warns users about wider spreads */}
                            {isExotic && (
                                <span
                                    className="chart-exotic-badge"
                                    title="Exotic pair — wider spreads, higher volatility, lower liquidity"
                                >
                                    Exotic ⚠
                                </span>
                            )}
                        </h2>
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
                            {/* Full label — hidden below 600 px via CSS */}
                            <span className="chart-a11y-full">
                                {useAccessiblePalette ? 'Use Standard Candles' : 'Use Color-Safe Candles'}
                            </span>
                            {/* Short label — shown below 600 px via CSS */}
                            <span className="chart-a11y-short">
                                {useAccessiblePalette ? 'Standard' : 'Color-Safe'}
                            </span>
                        </button>
                        <div className="chart-price">
                            {currentPrice === null ? '--' : currentPrice.toFixed(pairDecimals)}
                        </div>
                        {currentPrice === null ? (
                            <div className="chart-change">No market data</div>
                        ) : (
                            <div className={`chart-change ${isPositive ? 'positive' : 'negative'}`}>
                                {isPositive ? '▲ ' : '▼ '}
                                {isPositive ? '+' : ''}{priceChange.toFixed(pairDecimals)}{' '}
                                ({isPositive ? '+' : ''}{priceChangePct.toFixed(2)}%)
                            </div>
                        )}
                    </div>
                </div>
            </div>

            {/* ── Candlestick chart ── */}
            <div
                className="chart-container"
                ref={chartContainerRef}
                role="img"
                aria-label={`${pair} candlestick chart`}
            />

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
                        <button
                            className="date-input date-input-btn"
                            title="Open calendar"
                            onClick={() => dateInputRef.current?.showPicker()}
                        >
                            <Calendar size={14} />
                            <span>{selectedDate ? formatDisplayDate(selectedDate) : 'Pick a date'}</span>
                        </button>
                        <input
                            ref={dateInputRef}
                            type="date"
                            style={{ position: 'absolute', opacity: 0, pointerEvents: 'none', width: 0, height: 0 }}
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

                {/* Quick shortcuts */}
                <div className="date-shortcuts">
                    <span className="date-shortcuts-label">Quick select:</span>
                    {[
                        { label: '2022', date: '2022-01-03' },
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
            </div>

        </div>
    );
}