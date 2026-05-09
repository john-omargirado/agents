import axios from 'axios';

const API_BASE = '/api';
const api = axios.create({
    baseURL: API_BASE,
    timeout: 300000,
    headers: {
        ...(import.meta.env.DEV && {
            'X-API-KEY': import.meta.env.VITE_API_KEY,
        }),
    },
});

/**
 * Run the live MAS pipeline: [TTS + CE] → SIV → Verdict
 */
export async function runOrchestrator({
    pair,
    amount,
    leverage,
    riskThreshold,
    targetDate = null,
    experienceLevel = null,
} = {}) {
    const { data } = await api.post('/analyze', {
        currency_pair: pair,
        skip_llm: false,
        target_date: targetDate || null,
        live_mode: true,
        backtest_mode: false,
        accountCapital: amount,
        leverage,
        riskThreshold,
        experience_level: experienceLevel,
    });
    return data;
}

/**
 * Send a chat message to the live trading assistant.
 */
export async function sendChatMessage(
    message,
    currencyPair = null,
    analysisId = null,
    history = [],
    experienceLevel = null,
    simResult = null,
) {
    const { data } = await api.post('/chat', {
        message,
        currency_pair: currencyPair,
        analysis_id: analysisId,
        history,
        experience_level: experienceLevel,
        sim_result: simResult,
    });
    return data;
}

/**
 * Fetch recent run history for a pair.
 */
export async function getRunHistory(currencyPair, limit = 10) {
    const { data } = await api.get('/history', {
        params: { currency_pair: currencyPair, limit },
    });
    return data;
}

export async function getPairs() {
    const { data } = await api.get('/pairs');
    return data;
}

export async function getStrategies() {
    const { data } = await api.get('/strategies');
    return data;
}

export async function healthCheck() {
    const { data } = await api.get('/health');
    return data;
}

export async function getBacktestDates(currencyPair) {
    const { data } = await api.get('/backtest/dates', {
        params: { currency_pair: currencyPair },
    });
    return data;
}

export async function getBacktestNews(currencyPair, date = null) {
    const { data } = await api.get('/backtest/news', {
        params: {
            currency_pair: currencyPair,
            date: date || undefined,
        },
    });
    return data;
}

/**
 * Simulate a trade over the next 5 candles after the analysis date.
 */
export async function simulateTrade({
    currencyPair,
    action,         // 'BUY' | 'SELL' | 'HOLD'
    entryPrice,
    slDistance,
    tpDistance,
    targetDate,
}) {
    const { data } = await api.post('/simulate-trade', {
        currency_pair: currencyPair,
        action,
        entry_price: entryPrice,
        sl_distance: slDistance,
        tp_distance: tpDistance,
        target_date: targetDate || null,
    });
    return data;
}

/**
 * Run backtesting analysis for a specific date.
 */
export async function runBacktestAnalysis(
    currencyPair,
    date,
    skipLLM = false,
    profile = {}
) {
    const { data } = await api.post('/backtest/analyze', {
        currency_pair: currencyPair,
        date: date,
        skip_llm: skipLLM,
        live_mode: false,
        backtest_mode: true,
        accountCapital: profile.accountCapital,
        leverage: profile.leverage,
        riskThreshold: profile.riskThreshold,
    });
    return data;
}

export default api;