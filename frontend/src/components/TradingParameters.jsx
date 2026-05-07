import React, { useState } from 'react';
import { TrendingUp, DollarSign, Scaling, AlertTriangle, Info } from 'lucide-react';

const PAIRS = [
    // Major pairs — most liquid, tightest spreads, best for beginners
    { value: 'EUR/USD', label: 'EUR/USD', category: 'Major' },
    { value: 'USD/JPY', label: 'USD/JPY', category: 'Major' },
    { value: 'GBP/USD', label: 'GBP/USD', category: 'Major' },
    { value: 'AUD/USD', label: 'AUD/USD', category: 'Major' },
    { value: 'USD/CAD', label: 'USD/CAD', category: 'Major' },
    { value: 'USD/CHF', label: 'USD/CHF', category: 'Major' },
    // PHP pairs — exotic, wider spreads, higher risk
    { value: 'USD/PHP', label: 'USD/PHP', category: 'Exotic' },
    { value: 'EUR/PHP', label: 'EUR/PHP', category: 'Exotic' },
    { value: 'JPY/PHP', label: 'JPY/PHP', category: 'Exotic' },
];

const LEVERAGE_OPTIONS = [
    { value: '1:1', label: '1:1  — No leverage (safest)', risk: 'safe' },
    { value: '1:2', label: '1:2  — Very low risk', risk: 'safe' },
    { value: '1:5', label: '1:5  — Low risk (beginner)', risk: 'safe' },
    { value: '1:10', label: '1:10 — Moderate risk', risk: 'moderate' },
    { value: '1:20', label: '1:20 — High risk', risk: 'high' },
    { value: '1:50', label: '1:50 — Very high risk ⚠', risk: 'danger' },
    { value: '1:100', label: '1:100 — Extreme risk ⚠⚠', risk: 'danger' },
];

function Tooltip({ text }) {
    const [visible, setVisible] = useState(false);

    return (
        <span
            className="tooltip-wrap"
            onMouseEnter={() => setVisible(true)}
            onMouseLeave={() => setVisible(false)}
            onFocus={() => setVisible(true)}
            onBlur={() => setVisible(false)}
            tabIndex={0}
        >
            <Info size={13} className="tooltip-icon" />
            {visible && <span className="tooltip-box">{text}</span>}
        </span>
    );
}

export default function TradingParameters({
    pair, setPair,
    amount, setAmount,
    leverage, setLeverage,
    riskThreshold, setRiskThreshold,
    onSubmit,
    loading,
}) {
    const handleSubmit = (e) => {
        e.preventDefault();
        onSubmit();
    };

    const selectedLev = LEVERAGE_OPTIONS.find((l) => l.value === leverage);
    const leverageRisk = selectedLev?.risk || 'safe';
    const leverageMultiplier = leverage ? parseInt(leverage.split(':')[1]) : 1;

    const exposure = amount && leverageMultiplier
        ? (parseFloat(amount) * leverageMultiplier).toLocaleString('en-US', { maximumFractionDigits: 0 })
        : null;

    const selectedPair = PAIRS.find((p) => p.value === pair);
    const isExotic = selectedPair?.category === 'Exotic';

    return (
        <form className="card" onSubmit={handleSubmit}>

            {/* HEADER */}
            <div className="card-header">
                <div className="card-header-icon">
                    <TrendingUp size={18} />
                </div>
                <div>
                    <h2>Trading Parameters</h2>
                    <p>Configure your trade setup and risk profile</p>
                </div>
            </div>

            {/* FOREX PAIR */}
            <div className="form-group">
                <label className="form-label">
                    <TrendingUp size={14} />
                    Forex Pair
                    <Tooltip text="Major pairs are more stable. Exotic pairs are volatile and riskier." />
                </label>

                <select
                    className="form-select"
                    value={pair}
                    onChange={(e) => setPair(e.target.value)}
                >
                    <optgroup label="Major Pairs">
                        {PAIRS.filter((p) => p.category === 'Major').map((p) => (
                            <option key={p.value} value={p.value}>{p.label}</option>
                        ))}
                    </optgroup>

                    <optgroup label="Exotic Pairs">
                        {PAIRS.filter((p) => p.category === 'Exotic').map((p) => (
                            <option key={p.value} value={p.value}>{p.label}</option>
                        ))}
                    </optgroup>
                </select>

                <p className={`form-hint ${isExotic ? 'form-hint-warn' : ''}`}>
                    {isExotic
                        ? '⚠ Exotic pairs have higher volatility and wider spreads.'
                        : 'Major pairs are more liquid and beginner-friendly.'}
                </p>
            </div>

            {/* CAPITAL */}
            <div className="form-group">
                <label className="form-label">
                    <DollarSign size={14} />
                    Account Capital ($)
                    <Tooltip text="Your trading balance. Used to calculate risk and exposure." />
                </label>

                <input
                    type="number"
                    className="form-input"
                    value={amount}
                    onChange={(e) => setAmount(e.target.value)}
                    min="1"
                    placeholder="1000"
                />

                <p className="form-hint">
                    Total capital allocated for trading.
                </p>
            </div>

            {/* LEVERAGE */}
            <div className="form-group">
                <label className="form-label">
                    <Scaling size={14} />
                    Leverage
                    <Tooltip text="Higher leverage increases both profit potential and risk." />
                </label>

                <select
                    className={`form-select leverage-select leverage-${leverageRisk}`}
                    value={leverage}
                    onChange={(e) => setLeverage(e.target.value)}
                >
                    {LEVERAGE_OPTIONS.map((lev) => (
                        <option key={lev.value} value={lev.value}>
                            {lev.label}
                        </option>
                    ))}
                </select>

                {exposure && (
                    <div className={`leverage-exposure leverage-exposure-${leverageRisk}`}>
                        <span>
                            With ${Number(amount).toLocaleString()} at {leverage},
                            you control <strong>${exposure}</strong>.
                        </span>
                    </div>
                )}
            </div>

            {/* RISK */}
            <div className="form-group">
                <label className="form-label">
                    <AlertTriangle size={14} />
                    Max Risk Per Trade (%)
                    <Tooltip text="Professional traders risk 1–2% per trade." />
                </label>

                <input
                    type="number"
                    className="form-input"
                    value={riskThreshold}
                    onChange={(e) => setRiskThreshold(e.target.value)}
                    min="0.1"
                    max="100"
                    step="0.1"
                    placeholder="1"
                />

                {riskThreshold && (
                    <p className={`form-hint ${parseFloat(riskThreshold) > 5 ? 'form-hint-warn' : ''
                        }`}>
                        {parseFloat(riskThreshold) > 5
                            ? '⚠ High risk. Can quickly drain your account.'
                            : parseFloat(riskThreshold) <= 2
                                ? '✓ Ideal risk level.'
                                : 'Moderate risk. Keep it controlled.'}
                    </p>
                )}
            </div>

            {/* DISCLAIMER */}
            <p className="form-disclaimer">
                Educational tool only. Not financial advice.
            </p>

            {/* BUTTON */}
            <button
                type="submit"
                className="btn-primary"
                disabled={loading}
            >
                {loading ? (
                    <span className="loading-message">
                        <span className="spinner" />
                        Analyzing...
                    </span>
                ) : 'Get Analysis'}
            </button>
        </form>
    );
}