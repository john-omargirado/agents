// TradingParameters.jsx  (updated)
// Changes from original:
//  1. Every field now has a <ContextualHelp> button with plain-language explanations.
//  2. Experience level is surfaced as a visible pill bar (not buried in the tutorial modal).
//  3. A "Confirm before analysing" review screen appears before firing the API call.
//  4. Beginner-safe defaults: leverage defaults to 1:1, risk to 1%.
//  5. Field-level warnings are more prominent and explain *why*, not just *what*.

import React, { useState } from 'react';
import { TrendingUp, DollarSign, Scaling, AlertTriangle, CheckCircle, ChevronRight } from 'lucide-react';
import ContextualHelp from './ContextualHelp';



const PAIRS = [
    { value: 'EUR/USD', label: 'EUR/USD', category: 'Major' },
    { value: 'USD/JPY', label: 'USD/JPY', category: 'Major' },
    { value: 'GBP/USD', label: 'GBP/USD', category: 'Major' },
    { value: 'AUD/USD', label: 'AUD/USD', category: 'Major' },
    { value: 'USD/CAD', label: 'USD/CAD', category: 'Major' },
    { value: 'USD/CHF', label: 'USD/CHF', category: 'Major' },
    { value: 'USD/PHP', label: 'USD/PHP', category: 'Exotic' },
    { value: 'EUR/PHP', label: 'EUR/PHP', category: 'Exotic' },
    { value: 'JPY/PHP', label: 'JPY/PHP', category: 'Exotic' },
];

const LEVERAGE_OPTIONS = [
    { value: '1:1', label: '1:1  — No leverage (safest)', risk: 'safe' },
    { value: '1:2', label: '1:2  — Very low risk', risk: 'safe' },
    { value: '1:5', label: '1:5  — Low risk', risk: 'safe' },
    { value: '1:10', label: '1:10 — Moderate risk', risk: 'moderate' },
    { value: '1:20', label: '1:20 — High risk', risk: 'high' },
    { value: '1:50', label: '1:50 — Very high risk ⚠', risk: 'danger' },
    { value: '1:100', label: '1:100 — Extreme risk ⚠⚠', risk: 'danger' },
];

const LEVEL_OPTIONS = [
    { value: 'beginner', label: '🌱 New to forex' },
    { value: 'basic', label: '📈 Know the basics' },
    { value: 'intermediate', label: '⚙️ Experienced' },
];

const RISK_GUIDANCE = {
    beginner: {
        safe: '✓ Great choice. Risking 1–2% per trade is the professional standard.',
        moderate: '⚠ 3–5% is on the higher side. Consider dropping to 1–2% while learning.',
        high: '⚠ Above 5% can drain an account quickly after a few losses in a row.',
    },
    basic: {
        safe: '✓ 1–2% risk is solid money management.',
        moderate: 'Manageable, but watch your drawdown streak.',
        high: '⚠ High risk per trade — consider reducing.',
    },
    intermediate: {
        safe: '✓',
        moderate: 'Moderate.',
        high: '⚠ Above recommended threshold.',
    },
};

function getRiskLevel(val) {
    const n = parseFloat(val);
    if (!n || n <= 2) return 'safe';
    if (n <= 5) return 'moderate';
    return 'high';
}

// Confirm screen shown just before firing the analysis
function ConfirmScreen({ pair, amount, leverage, riskThreshold, experienceLevel, onBack, onConfirm, loading }) {
    const lvl = parseInt(leverage?.split(':')[1] || 1, 10);
    const exposure = amount ? (parseFloat(amount) * lvl).toLocaleString('en-US', { maximumFractionDigits: 0 }) : '—';
    const riskDollars = amount && riskThreshold
        ? ((parseFloat(amount) * parseFloat(riskThreshold)) / 100).toFixed(2)
        : '—';

    const isExotic = PAIRS.find(p => p.value === pair)?.category === 'Exotic';

    return (
        <div className="confirm-screen">
            <div className="confirm-header">
                <CheckCircle size={20} color="var(--buy-green)" />
                <div>
                    <div className="confirm-title">Ready to analyse</div>
                    <div className="confirm-sub">Here's what the system will use — check before running</div>
                </div>
            </div>

            <div className="confirm-grid">
                <div className="confirm-item">
                    <div className="ci-label">Pair</div>
                    <div className="ci-value">{pair}</div>
                    {isExotic && (
                        <div className="ci-warn">Exotic pair — wider spreads</div>
                    )}
                </div>
                <div className="confirm-item">
                    <div className="ci-label">Capital</div>
                    <div className="ci-value">${Number(amount).toLocaleString()}</div>
                </div>
                <div className="confirm-item">
                    <div className="ci-label">Leverage</div>
                    <div className="ci-value">{leverage}</div>
                    <div className="ci-sub">Controlling ${exposure}</div>
                </div>
                <div className="confirm-item">
                    <div className="ci-label">Max risk / trade</div>
                    <div className="ci-value">{riskThreshold}%</div>
                    <div className="ci-sub">${riskDollars} per trade</div>
                </div>
            </div>

            {experienceLevel === 'beginner' && (
                <div className="confirm-tip">
                    <span>🐢</span>
                    <span>
                        The system will analyse {pair} using charts, news, and a safety check — then explain everything in plain language.
                        No real money is involved.
                    </span>
                </div>
            )}

            <div className="confirm-actions">
                <button type="button" className="btn-ghost-sm" onClick={onBack}>
                    ← Edit setup
                </button>
                <button type="button" className="btn-primary" onClick={onConfirm} disabled={loading}>
                    {loading ? (
                        <>
                            <span className="spinner" />
                            Analysing…
                        </>
                    ) : (
                        <>
                            Run analysis <ChevronRight size={16} />
                        </>
                    )}
                </button>
            </div>
        </div>
    );
}

export default function TradingParameters({
    pair,
    setPair,
    amount,
    setAmount,
    leverage,
    setLeverage,
    riskThreshold,
    setRiskThreshold,
    onSubmit,
    loading,
    experienceLevel,
    setExperienceLevel,
}) {
    const [showConfirm, setShowConfirm] = useState(false);

    const handleReview = (e) => {
        e.preventDefault();
        setShowConfirm(true);
    };

    const handleConfirm = () => {
        setShowConfirm(false);
        onSubmit();
    };

    const selectedLev = LEVERAGE_OPTIONS.find(l => l.value === leverage);
    const leverageRisk = selectedLev?.risk || 'safe';
    const levMultiplier = leverage ? parseInt(leverage.split(':')[1]) : 1;
    const exposure = amount && levMultiplier
        ? (parseFloat(amount) * levMultiplier).toLocaleString('en-US', { maximumFractionDigits: 0 })
        : null;

    const riskLevel = getRiskLevel(riskThreshold);
    const riskGuidance = RISK_GUIDANCE[experienceLevel || 'beginner'];

    const selectedPairMeta = PAIRS.find(p => p.value === pair);
    const isExotic = selectedPairMeta?.category === 'Exotic';

    if (showConfirm) {
        return (
            <form className="card" onSubmit={e => e.preventDefault()}>
                <div className="card-header">
                    <div className="card-header-icon"><TrendingUp size={18} /></div>
                    <div><h2>Trading Parameters</h2><p>Review your setup</p></div>
                </div>
                <ConfirmScreen
                    pair={pair}
                    amount={amount}
                    leverage={leverage}
                    riskThreshold={riskThreshold}
                    experienceLevel={experienceLevel}
                    onBack={() => setShowConfirm(false)}
                    onConfirm={handleConfirm}
                    loading={loading}
                />
            </form>
        );
    }

    return (
        <form className="card" onSubmit={handleReview}>

            {/* HEADER */}
            <div className="card-header">
                <div className="card-header-icon"><TrendingUp size={18} /></div>
                <div>
                    <h2>Trading Parameters</h2>
                    <p>Configure your trade setup</p>
                </div>
            </div>

            {/* EXPERIENCE LEVEL — always visible, not buried in modal */}
            <div className="form-group">
                <label className="form-label">
                    Your experience level
                    <ContextualHelp topic={null} experienceLevel={experienceLevel} />
                </label>
                <div className="level-pills">
                    {LEVEL_OPTIONS.map(opt => (
                        <button
                            key={opt.value}
                            type="button"
                            className={`level-pill ${experienceLevel === opt.value ? 'active' : ''}`}
                            onClick={() => setExperienceLevel(opt.value)}
                        >
                            {opt.label}
                        </button>
                    ))}
                </div>
                {experienceLevel === 'beginner' && (
                    <p className="form-hint">
                        Every field will show a plain-language explanation when you click the <strong>?</strong> button.
                    </p>
                )}
            </div>

            {/* FOREX PAIR */}
            <div className="form-group">
                <label className="tp-label">
                    Forex Pair
                    <ContextualHelp
                        topic="pair"
                        experienceLevel={experienceLevel}
                    />
                </label>

                <select
                    className="form-select"
                    value={pair}
                    onChange={e => setPair(e.target.value)}
                >
                    <optgroup label="Major Pairs — recommended for beginners">
                        {PAIRS.filter(p => p.category === 'Major').map(p => (
                            <option key={p.value} value={p.value}>{p.label}</option>
                        ))}
                    </optgroup>
                    <optgroup label="Exotic Pairs — wider spreads, higher risk">
                        {PAIRS.filter(p => p.category === 'Exotic').map(p => (
                            <option key={p.value} value={p.value}>{p.label}</option>
                        ))}
                    </optgroup>
                </select>

                {isExotic ? (
                    <p className="form-hint form-hint-warn">
                        <AlertTriangle size={11} style={{ display: 'inline', marginRight: 4 }} />
                        Exotic pair — expect wider spreads, lower liquidity, and less reliable news data.
                        {experienceLevel === 'beginner' && ' Start with EUR/USD if you\'re just learning.'}
                    </p>
                ) : (
                    <p className="form-hint">Major pairs have the tightest spreads and most reliable data.</p>
                )}
            </div>

            {/* CAPITAL */}
            <div className="form-group">
                <label className="tp-label">
                    ($) Account Capital
                    <ContextualHelp
                        topic="capital"
                        experienceLevel={experienceLevel}
                    />
                </label>
                <input
                    type="number"
                    className="form-input"
                    value={amount}
                    onChange={e => setAmount(e.target.value)}
                    min="1"
                    placeholder="1000"
                />
                <p className="form-hint">
                    {experienceLevel === 'beginner'
                        ? 'This is your imaginary trading budget for the simulation — not real money.'
                        : 'Used to calculate position sizing and max dollar risk.'}
                </p>
            </div>

            {/* LEVERAGE */}
            <div className="form-group">
                <label className="tp-label">
                    Leverage
                    <ContextualHelp
                        topic="leverage"
                        experienceLevel={experienceLevel}
                    />
                </label>
                <select
                    className={`form-select leverage-select leverage-${leverageRisk}`}
                    value={leverage}
                    onChange={e => setLeverage(e.target.value)}
                >
                    {LEVERAGE_OPTIONS.map(lev => (
                        <option key={lev.value} value={lev.value}>{lev.label}</option>
                    ))}
                </select>

                {exposure && (
                    <div className={`leverage-exposure leverage-exposure-${leverageRisk}`}>
                        With ${Number(amount).toLocaleString()} at {leverage}, you control <strong>${exposure}</strong>.
                        {leverageRisk === 'danger' && experienceLevel === 'beginner' && (
                            <span> — For learning, 1:1 is much safer.</span>
                        )}
                    </div>
                )}
            </div>

            {/* RISK */}
            <div className="form-group">
                <label className="tp-label">
                    <AlertTriangle size={14} />
                    Max Risk Per Trade (%)
                    <ContextualHelp topic="risk" experienceLevel={experienceLevel} />
                </label>
                <input
                    type="number"
                    className="form-input"
                    value={riskThreshold}
                    onChange={e => setRiskThreshold(e.target.value)}
                    min="0.1"
                    max="100"
                    step="0.1"
                    placeholder="1"
                />
                {riskThreshold && (
                    <p className={`form-hint ${riskLevel !== 'safe' ? 'form-hint-warn' : ''}`}>
                        {riskGuidance?.[riskLevel] || ''}
                    </p>
                )}
            </div>

            <p className="form-disclaimer">Educational tool only. Not financial advice.</p>

            {/* Review button — leads to confirm screen, not direct submit */}
            <button type="submit" className="btn-primary" disabled={loading}>
                {loading ? (
                    <span className="loading-message">
                        <span className="spinner" />
                        Analysing…
                    </span>
                ) : (
                    <>
                        Review &amp; Analyse <ChevronRight size={16} />
                    </>
                )}
            </button>
        </form>
    );
}