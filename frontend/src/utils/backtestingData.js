/**
 * Fake backtesting data generator
 * Provides synthetic historical strategy performance data
 */

export const BACKTEST_PAIRS = [
    { value: 'EUR/USD', label: 'EUR/USD' },
    { value: 'USD/JPY', label: 'USD/JPY' },
    { value: 'GBP/USD', label: 'GBP/USD' },
    { value: 'AUD/USD', label: 'AUD/USD' },
    { value: 'USD/PHP', label: 'USD/PHP' },
];

/**
 * Generate fake backtesting results for a given pair and day
 * Simulates a 64-day backtest period with multiple trade signals
 */
export function generateBacktestDay(pair, dayNumber, totalDays = 64) {
    const seed = `${pair}-day-${dayNumber}`;
    const seedNum = seed.split('').reduce((acc, c) => acc + c.charCodeAt(0), 0);

    const random = () => {
        const x = Math.sin(seedNum * dayNumber + Math.random()) * 10000;
        return x - Math.floor(x);
    };

    // Base price depends on pair
    const basePrices = {
        'EUR/USD': 1.085,
        'USD/JPY': 145.50,
        'GBP/USD': 1.275,
        'AUD/USD': 0.685,
        'USD/PHP': 56.50,
    };

    const basePrice = basePrices[pair] || 1.085;
    const volatility = 0.002 + random() * 0.003;
    const direction = random() > 0.5 ? 1 : -1;

    // Generate candlestick data
    const open = basePrice + (random() - 0.5) * volatility;
    const close = open + direction * volatility;
    const high = Math.max(open, close) + random() * 0.001;
    const low = Math.min(open, close) - random() * 0.001;

    // Generate signal (BUY/SELL/HOLD)
    const signals = ['BUY', 'SELL', 'HOLD'];
    const signal = signals[Math.floor(random() * signals.length)];

    // Entry/Exit prices
    const entry = open + (random() - 0.5) * 0.002;
    const exit = close + (random() - 0.5) * 0.002;

    // Calculate pips (difference in 4th decimal for major pairs)
    const pipsPerPoint = pair.includes('JPY') ? 100 : 10000;
    const pips = Math.round((exit - entry) * pipsPerPoint);

    // P&L (assuming $1000 account and 1:1 leverage)
    const pnl = (pips / pipsPerPoint) * 1000 * (signal === 'HOLD' ? 0 : 1);

    return {
        dayNumber,
        totalDays,
        pair,
        date: new Date(2024, 11, 31 - (totalDays - dayNumber)),
        ohlcv: [
            {
                time: Math.floor(new Date(2024, 11, 31 - (totalDays - dayNumber)).getTime() / 1000),
                open: Number(open.toFixed(5)),
                high: Number(high.toFixed(5)),
                low: Number(low.toFixed(5)),
                close: Number(close.toFixed(5)),
            },
        ],
        signal: signal,
        entry: Number(entry.toFixed(5)),
        exit: Number(exit.toFixed(5)),
        pips: pips,
        pnl: Number(pnl.toFixed(2)),
        win: pnl > 0,
        tradeTaken: signal !== 'HOLD',
    };
}

/**
 * Generate cumulative backtesting stats across all days
 */
export function generateBacktestStats(pair, totalDays = 64) {
    let totalPnl = 0;
    let wins = 0;
    let losses = 0;
    let holds = 0;
    let totalTrades = 0;

    for (let day = 1; day <= totalDays; day++) {
        const dayData = generateBacktestDay(pair, day, totalDays);
        totalPnl += dayData.pnl;

        if (dayData.tradeTaken) {
            totalTrades++;
            if (dayData.win) {
                wins++;
            } else {
                losses++;
            }
        } else {
            holds++;
        }
    }

    const winRate = totalTrades > 0 ? ((wins / totalTrades) * 100).toFixed(1) : 0;

    return {
        totalPnl: Number(totalPnl.toFixed(2)),
        wins,
        losses,
        holds,
        totalTrades,
        winRate: parseFloat(winRate),
    };
}
