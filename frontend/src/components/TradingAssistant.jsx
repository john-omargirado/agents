import React, { useState, useRef, useEffect } from 'react';
import {
    Bot, Send, TrendingUp, TrendingDown, Minus,
    BarChart3, Target, ShieldCheck, RefreshCw, Activity, User,
    History, X, ChevronRight, AlertTriangle, Info,
    Play, Pause, CheckCircle, XCircle, Clock,
} from 'lucide-react';
import { sendChatMessage, simulateTrade } from '../services/api';
import TradingTurtleMascot from './TradingTurtleMascot';

// ─── Helpers ──────────────────────────────────────────────────────────────────

function formatBoldText(text) {
    if (!text) return text;
    const parts = text.split(/(\*\*[^*]+\*\*)/);
    return parts.map((part, i) => {
        if (part.startsWith('**') && part.endsWith('**')) {
            return <strong key={i}>{part.slice(2, -2)}</strong>;
        }
        return part;
    });
}

function SignalBadge({ signal }) {
    const s = (signal || 'HOLD').toUpperCase();
    const cls = s === 'BUY' ? 'buy' : s === 'SELL' ? 'sell' : 'hold';
    const Icon = s === 'BUY' ? TrendingUp : s === 'SELL' ? TrendingDown : Minus;
    return (
        <span className={`signal-badge ${cls}`}>
            <Icon size={16} /> {s}
        </span>
    );
}

function getPipSize(currencyPair) {
    return (currencyPair || '').toUpperCase().includes('JPY') ? 0.01 : 0.0001;
}

function toPips(priceDiff, currencyPair) {
    const isJPY = (currencyPair || '').toUpperCase().includes('JPY');
    return (Math.abs(priceDiff) / (isJPY ? 0.01 : 0.0001)).toFixed(1);
}

function AgentDetail({ label, value, extra }) {
    const toneClass =
        value === 'BUY' || value === 'STRONG' || value === 'COHERENT' ? 'positive'
            : value === 'SELL' || value === 'WEAK' || value === 'INCOHERENT' ? 'negative'
                : 'neutral';
    return (
        <div className="agent-detail-row">
            <span className="agent-detail-label">{label}</span>
            <span className="agent-detail-value-wrap">
                <span className={`agent-detail-value ${toneClass}`}>{value}</span>
                {extra && <span className="agent-detail-extra">{extra}</span>}
            </span>
        </div>
    );
}

function SIVAuditRow({ issues }) {
    const [expanded, setExpanded] = React.useState(false);
    const issueLabels = {
        signal_mismatch: 'CE and TTS point in opposite directions',
        one_signal_neutral: 'One agent returned neutral — partial agreement only',
        missing_price: 'Price data missing — cannot verify signal integrity',
        price_mismatch: 'CE and TTS used different prices — data inconsistency',
        price_parse_error: 'Price data could not be parsed',
        unrecognized_signal: 'One agent returned an unrecognized signal value',
    };
    const labeled = issues.map(i => issueLabels[i] || i);
    const preview = labeled[0]?.slice(0, 50) + (labeled[0]?.length > 50 ? '...' : '');
    return (
        <div className="agent-detail-row agent-detail-issues">
            <span className="agent-detail-label">SIV Audit</span>
            <span
                className="agent-detail-value siv-audit-value"
                onClick={() => setExpanded(!expanded)}
                title="Click to expand"
                style={{ cursor: 'pointer' }}
            >
                {expanded
                    ? labeled.map((issue, i) => <span key={i} className="siv-audit-issue">{issue}</span>)
                    : <span className="siv-audit-issue">
                        {preview}{' '}
                        {labeled.length > 1 && <span className="siv-audit-count">+{labeled.length - 1} more</span>}
                    </span>
                }
            </span>
        </div>
    );
}

function SkipExplanation({ reasoning, ce, tts, verdict }) {
    if (!reasoning) return null;
    const r = reasoning.toLowerCase();
    const ceScore = Math.abs(Number(ce?.ce_score || 0));
    const ttsScore = Math.abs(Number(tts?.total_score || 0));
    const articles = Number(ce?.article_count || 0);
    const wscore = Number(verdict?.weighted_score || 0);

    let title = '';
    let body = '';
    let borderColor = 'var(--signal-hold-border)';
    let bg = 'var(--signal-hold-bg)';
    let titleColor = 'var(--hold-yellow)';

    if (r.includes('incoherent')) {
        title = '🚫 Price integrity check failed (SIV blocked)';
        body = "The price seen by the news agent (CE) didn't match the price from the technical agent (TTS). "
            + "The system blocked the trade to avoid acting on inconsistent data. "
            + "This is a data safety block, not a weak signal.";
        borderColor = 'var(--signal-sell-border)';
        bg = 'var(--signal-sell-bg)';
        titleColor = 'var(--sell-red)';
    } else if (r.includes('signal_mismatch')) {
        title = '⚠️ News and technical signals disagree';
        body = `CE (news sentiment) and TTS (technical analysis) pointed in opposite directions. `
            + `When signals conflict, the system only proceeds if news is strong enough: `
            + `needs 10+ articles with |ce_score| ≥ 0.05.\n\n`
            + `This run: ${articles} article(s), |ce_score| = ${ceScore.toFixed(3)} — threshold not met.`;
    } else if (r.includes('weak signals')) {
        const ttsHit = ttsScore >= 0.08;
        const ceHit = ceScore >= 0.05;
        const artHit = articles >= 10;
        title = '📉 Signals too weak to trade';
        body = `Both the technical score and the news score are below the minimum thresholds. `
            + `The system needs at least one strong signal to act:\n\n`
            + `  ${ttsHit ? '✓' : '✗'} TTS score: ${ttsScore.toFixed(3)} (needs ≥ 0.08)\n`
            + `  ${ceHit ? '✓' : '✗'} CE score:  ${ceScore.toFixed(3)}  (needs ≥ 0.05)\n`
            + `  ${artHit ? '✓' : '✗'} Articles:  ${articles}            (needs ≥ 10 for CE to count as strong)\n\n`
            + `Note: A high CE score with few articles still fails — more articles = more trustworthy sentiment.`;
    } else if (r.includes('atr') || r.includes('invalid sl')) {
        title = '⚠️ Volatility too low to size a position';
        body = 'ATR was near zero. Without a measurable volatility range, the system cannot calculate a safe stop-loss distance, so no trade was placed.';
    } else if (verdict?.decision === 'HOLD' && Math.abs(wscore) < 0.10) {
        title = '⏸ Score below action threshold';
        body = `The combined score (${wscore >= 0 ? '+' : ''}${wscore.toFixed(4)}) didn't reach the ±0.10 threshold needed for a BUY or SELL. `
            + `Signals exist but aren't strong or aligned enough yet.`;
    } else {
        title = 'ℹ️ No trade taken this run';
        body = reasoning;
    }

    return (
        <div style={{
            margin: '10px 0', padding: '12px 14px',
            borderRadius: 'var(--radius-sm)',
            border: `1px solid ${borderColor}`, background: bg,
            fontSize: 12, lineHeight: 1.75,
        }}>
            <div style={{ fontWeight: 700, color: titleColor, marginBottom: 6, fontSize: 13 }}>{title}</div>
            <div style={{ color: 'var(--text-secondary)', whiteSpace: 'pre-line' }}>{body}</div>
        </div>
    );
}

function EnsembleBreakdown({ ce, tts, siv, verdict }) {
    const ce_conf = Number(ce?.ce_confidence || 0);
    const ce_w = Math.max(0, Math.min(1, 0.35 + 0.30 * ce_conf));
    const tts_w = 1 - ce_w;
    const ce_pct = Math.round(ce_w * 100);
    const tts_pct = Math.round(tts_w * 100);
    const siv_signal = (siv?.signal || '').toUpperCase();
    const siv_mult = Number(siv?.score_multiplier ?? 1);
    const wscore = Number(verdict?.weighted_score || 0);
    const articles = Number(ce?.article_count || 0);

    const confLabel = articles >= 25 ? 'HIGH' : articles >= 15 ? 'MODERATE' : 'LOW';
    const confColor = articles >= 25 ? 'var(--buy-green)' : articles >= 15 ? 'var(--hold-yellow)' : 'var(--text-muted)';
    const wColor = wscore >= 0.10 ? 'var(--buy-green)' : wscore <= -0.10 ? 'var(--sell-red)' : 'var(--hold-yellow)';

    return (
        <div style={{
            background: 'var(--bg-input)', border: '1px solid var(--bg-input-border)',
            borderRadius: 'var(--radius-sm)', padding: '10px 12px', marginBottom: 10, fontSize: 12,
        }}>
            <div style={{ fontWeight: 600, color: 'var(--text-muted)', marginBottom: 8, fontSize: 11, textTransform: 'uppercase', letterSpacing: '0.5px' }}>
                How this verdict was scored
            </div>
            <div style={{ display: 'flex', height: 6, borderRadius: 4, overflow: 'hidden', marginBottom: 6, gap: 2 }}>
                <div style={{ width: `${tts_pct}%`, background: 'var(--accent)', borderRadius: 4 }} />
                <div style={{ width: `${ce_pct}%`, background: '#818cf8', borderRadius: 4 }} />
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 8, fontSize: 11 }}>
                <span style={{ color: 'var(--accent)' }}>▪ Technical (TTS) {tts_pct}%</span>
                <span style={{ color: '#818cf8' }}>▪ News (CE) {ce_pct}%</span>
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 6, marginBottom: 8 }}>
                <div style={{ background: 'var(--bg-card)', borderRadius: 6, padding: '6px 8px' }}>
                    <div style={{ color: 'var(--text-muted)', fontSize: 10, marginBottom: 2 }}>TTS score</div>
                    <div style={{ fontWeight: 700, color: 'var(--accent)', fontVariantNumeric: 'tabular-nums' }}>
                        {Number(tts?.total_score || 0) >= 0 ? '+' : ''}{Number(tts?.total_score || 0).toFixed(4)}
                    </div>
                </div>
                <div style={{ background: 'var(--bg-card)', borderRadius: 6, padding: '6px 8px' }}>
                    <div style={{ color: 'var(--text-muted)', fontSize: 10, marginBottom: 2 }}>CE score</div>
                    <div style={{ fontWeight: 700, color: '#818cf8', fontVariantNumeric: 'tabular-nums' }}>
                        {Number(ce?.ce_score || 0) >= 0 ? '+' : ''}{Number(ce?.ce_score || 0).toFixed(4)}
                    </div>
                </div>
            </div>
            <div style={{ color: 'var(--text-secondary)', lineHeight: 1.65, fontSize: 12 }}>
                CE carries <strong>{ce_pct}%</strong> weight because news confidence is{' '}
                <span style={{ color: confColor, fontWeight: 600 }}>{confLabel}</span>
                {' '}({articles} article{articles !== 1 ? 's' : ''} found).
                {' '}{siv_signal === 'COHERENT'
                    ? 'SIV confirmed both signals agree — full score applied.'
                    : siv_signal === 'INCOHERENT'
                        ? 'SIV flagged a price mismatch — score zeroed out entirely.'
                        : `SIV applied a ×${siv_mult.toFixed(2)} penalty (signals partially misaligned).`}
                {' '}Final score:{' '}
                <strong style={{ color: wColor, fontVariantNumeric: 'tabular-nums' }}>
                    {wscore >= 0 ? '+' : ''}{wscore.toFixed(4)}
                </strong>
                {Math.abs(wscore) < 0.001 && (
                    <span style={{ color: 'var(--text-muted)', fontSize: 11 }}> (no signal)</span>
                )}
            </div>
        </div>
    );
}

// ─── Trade Action Buttons ─────────────────────────────────────────────────────
function TradeActionButtons({ onAction, disabled }) {
    return (
        <div style={{
            margin: '5px 10px 10px',
            padding: '10px 12px',
            background: 'var(--bg-input)',
            border: '1px solid var(--bg-input-border)',
            borderRadius: 'var(--radius-sm)',
        }}>
            <div style={{
                fontSize: 12, fontWeight: 600,
                color: 'var(--text-secondary)',
                marginBottom: 10,
                display: 'flex', alignItems: 'center', gap: 6,
            }}>
                <Play size={12} style={{ color: 'var(--accent)' }} />
                What would you like to do?
            </div>
            <div style={{ display: 'flex', gap: 8 }}>
                {/* BUY */}
                <button
                    onClick={() => onAction('BUY')}
                    disabled={disabled}
                    style={{
                        flex: 1, padding: '9px 0',
                        borderRadius: 8,
                        border: '1px solid rgba(46,168,74,0.5)',
                        background: 'rgba(46,168,74,0.12)',
                        color: '#2EA84A',
                        fontWeight: 700, fontSize: 13,
                        cursor: disabled ? 'not-allowed' : 'pointer',
                        display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 5,
                        transition: 'all 0.15s ease',
                        opacity: disabled ? 0.5 : 1,
                    }}
                    onMouseEnter={e => { if (!disabled) e.currentTarget.style.background = 'rgba(46,168,74,0.22)'; }}
                    onMouseLeave={e => { if (!disabled) e.currentTarget.style.background = 'rgba(46,168,74,0.12)'; }}
                >
                    <TrendingUp size={14} /> BUY
                </button>

                {/* SELL */}
                <button
                    onClick={() => onAction('SELL')}
                    disabled={disabled}
                    style={{
                        flex: 1, padding: '9px 0',
                        borderRadius: 8,
                        border: '1px solid rgba(220,53,69,0.5)',
                        background: 'rgba(220,53,69,0.12)',
                        color: '#dc3545',
                        fontWeight: 700, fontSize: 13,
                        cursor: disabled ? 'not-allowed' : 'pointer',
                        display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 5,
                        transition: 'all 0.15s ease',
                        opacity: disabled ? 0.5 : 1,
                    }}
                    onMouseEnter={e => { if (!disabled) e.currentTarget.style.background = 'rgba(220,53,69,0.22)'; }}
                    onMouseLeave={e => { if (!disabled) e.currentTarget.style.background = 'rgba(220,53,69,0.12)'; }}
                >
                    <TrendingDown size={14} /> SELL
                </button>

                {/* HOLD */}
                <button
                    onClick={() => onAction('HOLD')}
                    disabled={disabled}
                    style={{
                        flex: 1, padding: '9px 0',
                        borderRadius: 8,
                        border: '1px solid rgba(234,179,8,0.4)',
                        background: 'rgba(234,179,8,0.08)',
                        color: '#d4a017',
                        fontWeight: 700, fontSize: 13,
                        cursor: disabled ? 'not-allowed' : 'pointer',
                        display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 5,
                        transition: 'all 0.15s ease',
                        opacity: disabled ? 0.5 : 1,
                    }}
                    onMouseEnter={e => { if (!disabled) e.currentTarget.style.background = 'rgba(234,179,8,0.15)'; }}
                    onMouseLeave={e => { if (!disabled) e.currentTarget.style.background = 'rgba(234,179,8,0.08)'; }}
                >
                    <Minus size={14} /> HOLD
                </button>
            </div>
            <div style={{ fontSize: 10, color: 'var(--text-muted)', marginTop: 8, textAlign: 'center' }}>
                Simulates what would have happened over the next 5 candles
            </div>
        </div>
    );
}

// ─── Candle Timeline ──────────────────────────────────────────────────────────
function CandleTimeline({ candles, action, entryPrice, tpPrice, slPrice, exitCandle, pair }) {
    if (!candles || candles.length === 0) return null;

    const pip = (pair || '').toUpperCase().includes('JPY') ? 0.01 : 0.0001;

    return (
        <div style={{ marginTop: 10 }}>
            <div style={{ fontSize: 10, fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.5px', marginBottom: 6 }}>
                5-Candle Replay
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
                {candles.map((c, i) => {
                    const isExit = i + 1 === exitCandle;
                    const isBuy = action === 'BUY';
                    const tpHit = isBuy ? c.high >= tpPrice : c.low <= tpPrice;
                    const slHit = isBuy ? c.low <= slPrice : c.high >= slPrice;
                    const change = ((c.close - c.open) / pip).toFixed(1);
                    const isUp = c.close >= c.open;

                    return (
                        <div key={i} style={{
                            display: 'flex', alignItems: 'center', gap: 8,
                            padding: '5px 8px',
                            borderRadius: 6,
                            background: isExit
                                ? tpHit ? 'rgba(46,168,74,0.1)' : slHit ? 'rgba(220,53,69,0.1)' : 'rgba(234,179,8,0.08)'
                                : 'var(--bg-input)',
                            border: isExit
                                ? tpHit ? '1px solid rgba(46,168,74,0.3)' : slHit ? '1px solid rgba(220,53,69,0.3)' : '1px solid rgba(234,179,8,0.2)'
                                : '1px solid transparent',
                            fontSize: 11,
                        }}>
                            <span style={{ color: 'var(--text-muted)', width: 48, flexShrink: 0 }}>
                                C{i + 1} {c.date ? `(${c.date.slice(5)})` : ''}
                            </span>
                            <span style={{ color: isUp ? '#2EA84A' : '#dc3545', width: 16, textAlign: 'center' }}>
                                {isUp ? '▲' : '▼'}
                            </span>
                            <span style={{ color: 'var(--text-secondary)', fontVariantNumeric: 'tabular-nums', flex: 1 }}>
                                O:{c.open?.toFixed(4)} H:{c.high?.toFixed(4)} L:{c.low?.toFixed(4)} C:{c.close?.toFixed(4)}
                            </span>
                            <span style={{ color: isUp ? '#2EA84A' : '#dc3545', fontSize: 10, fontVariantNumeric: 'tabular-nums' }}>
                                {isUp ? '+' : ''}{change}p
                            </span>
                            {isExit && (
                                <span style={{
                                    fontSize: 10, fontWeight: 700, padding: '1px 5px', borderRadius: 4,
                                    background: tpHit ? 'rgba(46,168,74,0.2)' : slHit ? 'rgba(220,53,69,0.2)' : 'rgba(234,179,8,0.15)',
                                    color: tpHit ? '#2EA84A' : slHit ? '#dc3545' : '#d4a017',
                                    flexShrink: 0,
                                }}>
                                    {tpHit ? '✓ TP' : slHit ? '✗ SL' : '⏱ EXIT'}
                                </span>
                            )}
                        </div>
                    );
                })}
            </div>
        </div>
    );
}

// ─── Simulation Result Card ───────────────────────────────────────────────────
function SimTradeResult({ simResult, analysisResult, experienceLevel = 'beginner' }) {
    if (!simResult) return null;

    const { outcome, action, entry_price, exit_price, exit_candle,
        pnl_pips, tp_price, sl_price, candles, pair, message } = simResult;

    if (outcome === 'HOLD') {
        return (
            <div style={{
                margin: '10px 14px', padding: '10px 12px',
                borderRadius: 'var(--radius-sm)',
                border: '1px solid rgba(234,179,8,0.3)',
                background: 'rgba(234,179,8,0.06)', fontSize: 12,
            }}>
                <div style={{ fontWeight: 700, color: '#d4a017', marginBottom: 4, fontSize: 13 }}>
                    ⏸ You chose to HOLD
                </div>
                <div style={{ color: 'var(--text-secondary)' }}>{message}</div>
            </div>
        );
    }

    const isWin = outcome === 'TAKE_PROFIT';
    const isLoss = outcome === 'STOP_LOSS';
    const pnlColor = pnl_pips > 0 ? '#2EA84A' : pnl_pips < 0 ? '#dc3545' : 'var(--text-muted)';
    const borderColor = isWin ? 'rgba(46,168,74,0.4)' : isLoss ? 'rgba(220,53,69,0.4)' : 'rgba(234,179,8,0.3)';
    const bgColor = isWin ? 'rgba(46,168,74,0.06)' : isLoss ? 'rgba(220,53,69,0.06)' : 'rgba(234,179,8,0.05)';
    const outcomeLabel = isWin ? '✅ Take Profit Hit' : isLoss ? '❌ Stop Loss Hit' : '⏱ Time Exit (5 candles)';
    const outcomeColor = isWin ? '#2EA84A' : isLoss ? '#dc3545' : '#d4a017';

    const getNarrative = () => {
        const lvl = experienceLevel || 'beginner';
        const pips = Math.abs(pnl_pips ?? 0).toFixed(1);
        const pnlSigned = `${(pnl_pips ?? 0) >= 0 ? '+' : ''}${(pnl_pips ?? 0).toFixed(1)}`;

        if (isWin) {
            if (lvl === 'intermediate')
                return `${action} closed TP on C${exit_candle}. +${pips} pips.`;
            if (lvl === 'basic')
                return `The ${action} trade hit take profit on candle ${exit_candle}, banking ${pips} pips. Price moved cleanly in the expected direction without triggering the stop.`;
            return `Your ${action} trade worked out — price reached the take profit target on candle ${exit_candle} and closed in profit. You gained ${pips} pips. A pip is just a tiny unit of price movement, so gaining ${pips} of them means price traveled that far in your favor. Nice one!`;
        }

        if (isLoss) {
            if (lvl === 'intermediate')
                return `${action} stopped out on C${exit_candle}. −${pips} pips.`;
            if (lvl === 'basic')
                return `Stop loss triggered on candle ${exit_candle}, costing ${pips} pips. The SL did its job — it capped the loss and prevented further drawdown.`;
            return `Price moved against the trade and hit the stop loss on candle ${exit_candle}. You lost ${pips} pips — but that's exactly what the stop loss (SL) is there for. It's a safety net that automatically closes the trade to limit how much you can lose, no matter how far price keeps moving.`;
        }

        // time exit
        if (lvl === 'intermediate')
            return `Time exit C${exit_candle}. ${pnlSigned} pips at ${exit_price?.toFixed(4)}.`;
        if (lvl === 'basic')
            return `Neither TP nor SL was hit within 5 candles. Trade closed at candle ${exit_candle} for ${pnlSigned} pips. Price lacked follow-through in either direction.`;
        return `Neither the take profit nor the stop loss was reached within the 5 candles we followed. The trade was automatically closed at the end of candle ${exit_candle} with a result of ${pnlSigned} pips. This happens when price moves sideways or doesn't push far enough — it's a common outcome and totally normal.`;
    };

    return (
        <div style={{
            margin: '10px 14px', padding: '10px 12px',
            borderRadius: 'var(--radius-sm)',
            border: `1px solid ${borderColor}`,
            background: bgColor, fontSize: 12,
        }}>
            {/* Header */}
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 10 }}>
                <div>
                    <div style={{ fontWeight: 700, color: outcomeColor, fontSize: 13 }}>{outcomeLabel}</div>
                    <div style={{ color: 'var(--text-muted)', fontSize: 11, marginTop: 2 }}>
                        {action} · exited on candle {exit_candle}
                    </div>
                </div>
                <div style={{
                    textAlign: 'right', padding: '4px 10px',
                    background: (pnl_pips ?? 0) >= 0 ? 'rgba(46,168,74,0.15)' : 'rgba(220,53,69,0.15)',
                    borderRadius: 6,
                }}>
                    <div style={{ fontSize: 10, color: 'var(--text-muted)' }}>P&L</div>
                    <div style={{ fontWeight: 800, fontSize: 15, color: pnlColor, fontVariantNumeric: 'tabular-nums' }}>
                        {(pnl_pips ?? 0) >= 0 ? '+' : ''}{(pnl_pips ?? 0).toFixed(1)} pips
                    </div>
                </div>
            </div>

            {/* Price grid — 2×2 */}
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 6, marginBottom: 10 }}>
                {[
                    { label: 'Entry', value: entry_price?.toFixed(4), color: 'var(--text-secondary)' },
                    { label: 'Exit Price', value: exit_price?.toFixed(4), color: outcomeColor },
                    { label: 'Take Profit', value: tp_price?.toFixed(4), color: '#2EA84A' },
                    { label: 'Stop Loss', value: sl_price?.toFixed(4), color: '#dc3545' },
                ].map(({ label, value, color }) => (
                    <div key={label} style={{ background: 'var(--bg-input)', borderRadius: 6, padding: '5px 8px' }}>
                        <div style={{ fontSize: 10, color: 'var(--text-muted)', marginBottom: 1 }}>{label}</div>
                        <div style={{ fontWeight: 700, color, fontVariantNumeric: 'tabular-nums', fontSize: 12 }}>{value || '—'}</div>
                    </div>
                ))}
            </div>

            {/* Level-aware narrative */}
            <div style={{
                color: 'var(--text-secondary)', lineHeight: 1.7,
                padding: '8px 10px',
                background: 'var(--bg-input)',
                borderRadius: 6, marginBottom: 10,
            }}>
                {getNarrative()}
            </div>

            <CandleTimeline
                candles={candles}
                action={action}
                entryPrice={entry_price}
                tpPrice={tp_price}
                slPrice={sl_price}
                exitCandle={exit_candle}
                pair={pair || analysisResult?.currency_pair}
            />
        </div>
    );
}

// ─── Analysis Card ────────────────────────────────────────────────────────────
function AnalysisCard({ result, pairLabel }) {
    if (!result) return null;

    const ceNoNews =
        (result.ce?.article_count === 0) ||
        result.ce?.explanation === 'no_data' ||
        result.ce?.explanation === 'no_news';

    const sivIssues = result.siv?.issues || [];
    const trade = result.trade || {};
    const backendLotSize = trade.position_size ?? null;
    const entryPrice = result.tts?.price || 0;
    const isBuy = result.verdict?.decision === 'BUY';
    const action = (result.verdict?.action || '').toUpperCase();
    const decision = (result.verdict?.decision || 'HOLD').toUpperCase();
    const isSkipped = action === 'SKIP' || action === 'NONE' || decision === 'HOLD';

    const take_profit = entryPrice
        ? isBuy ? entryPrice + (trade.tp_distance || 0) : entryPrice - (trade.tp_distance || 0)
        : null;
    const stop_loss = entryPrice
        ? isBuy ? entryPrice - (trade.sl_distance || 0) : entryPrice + (trade.sl_distance || 0)
        : null;

    const rp = (!isSkipped && trade.sl_distance != null) ? {
        lot_size: trade.position_size,
        take_profit,
        stop_loss,
        risk_reward_ratio: trade.tp_distance && trade.sl_distance
            ? trade.tp_distance / trade.sl_distance
            : null,
    } : null;

    const retryCount = result.meta?.retry_count || 0;
    const notes = [];
    if (!isSkipped && backendLotSize == null) notes.push('Lot sizing unavailable — provide account capital and risk threshold.');
    if (!entryPrice) notes.push('Entry price missing — TTS price not computed or data is stale.');

    return (
        <div className="assistant-message">
            <div className="message-header">
                <span>Analysis for {result.currency_pair || pairLabel}</span>
                {result.target_date && (
                    <span className="message-date-tag">{result.target_date}</span>
                )}
            </div>

            <div className="signal-row">
                <SignalBadge signal={decision} />
                {isSkipped && (
                    <span style={{ fontSize: 12, color: 'var(--text-muted)', marginLeft: 6 }}>
                        No trade taken
                    </span>
                )}
            </div>

            {retryCount > 0 && (
                <div className="orchestrator-info">
                    <RefreshCw size={14} />
                    SIV validation required {retryCount} retry(s) before passing
                </div>
            )}

            {ceNoNews && (
                <div className="orchestrator-info orchestrator-info-warn">
                    <AlertTriangle size={14} />
                    No macroeconomic news found — CE sentiment is neutral, directional impact near zero.
                    {result.ce?.article_count != null && (
                        <span>
                            {' '}(articles: {result.ce.article_count}
                            {result.ce.raw_article_count != null ? ` / raw: ${result.ce.raw_article_count}` : ''}
                            {result.target_date ? `, date: ${result.target_date}` : ''})
                        </span>
                    )}
                </div>
            )}

            {isSkipped && (
                <SkipExplanation
                    reasoning={result.verdict?.verdict_reasoning}
                    ce={result.ce}
                    tts={result.tts}
                    verdict={result.verdict}
                />
            )}

            {!isSkipped && (
                <p className="analysis-text">
                    {result.verdict?.verdict_reasoning || result.tts?.explanation || 'No explanation available.'}
                </p>
            )}

            <EnsembleBreakdown ce={result.ce} tts={result.tts} siv={result.siv} verdict={result.verdict} />

            {!isSkipped && (
                <div className="stats-row">
                    <div className="stat-item">
                        <div className="stat-label"><BarChart3 size={12} /> Lot Size</div>
                        <div className="stat-value">
                            {typeof backendLotSize === 'number' ? backendLotSize.toFixed(3) : '—'}
                        </div>
                    </div>
                    <div className="stat-item">
                        <div className="stat-label"><Target size={12} /> Entry Price</div>
                        <div className="stat-value">{entryPrice ? entryPrice.toFixed(4) : '—'}</div>
                    </div>
                </div>
            )}

            <div className="quick-params">
                <div className="quick-item">
                    <strong>TTS</strong>: {result.tts?.decision || 'N/A'}
                    <span style={{ color: 'var(--text-muted)', fontSize: 11, marginLeft: 4 }}>
                        (technical analysis — score {Number(result.tts?.total_score || 0) >= 0 ? '+' : ''}{Number(result.tts?.total_score || 0).toFixed(3)})
                    </span>
                </div>
                <div className="quick-item">
                    <strong>CE</strong>: {result.ce?.sentiment || 'N/A'}
                    {ceNoNews
                        ? <em style={{ color: 'var(--text-muted)', fontSize: 11 }}> (no news data)</em>
                        : <span style={{ color: 'var(--text-muted)', fontSize: 11, marginLeft: 4 }}>
                            (news sentiment — {result.ce?.article_count || 0} article{result.ce?.article_count !== 1 ? 's' : ''})
                        </span>
                    }
                </div>
                <div className="quick-item">
                    <strong>SIV</strong>: {result.siv?.signal || 'N/A'}
                    <span style={{ color: 'var(--text-muted)', fontSize: 11, marginLeft: 4 }}>
                        (integrity check — ×{Number(result.siv?.score_multiplier ?? 1).toFixed(2)} multiplier)
                    </span>
                </div>
                <div className="quick-item">
                    <strong>Regime</strong>: {result.tts?.regime || 'N/A'}
                    <span style={{ color: 'var(--text-muted)', fontSize: 11, marginLeft: 4 }}>
                        {result.tts?.regime === 'TRENDING' ? '(trend signals weighted higher)'
                            : result.tts?.regime === 'RANGING' ? '(mean-reversion signals weighted higher)'
                                : result.tts?.regime === 'TRANSITIONAL' ? '(blended signal weights)' : ''}
                    </span>
                </div>
            </div>

            {notes.length > 0 && (
                <div className="analysis-notes">
                    <strong>Notes:</strong>
                    <ul>{notes.map((n, i) => <li key={i}>{n}</li>)}</ul>
                </div>
            )}

            {rp && (() => {
                const pairStr = result.currency_pair || pairLabel || '';
                const tpPips = entryPrice && rp.take_profit ? toPips(rp.take_profit - entryPrice, pairStr) : null;
                const slPips = entryPrice && rp.stop_loss ? toPips(rp.stop_loss - entryPrice, pairStr) : null;
                return (
                    <div className="risk-params">
                        <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 6 }}>
                            SL (Stop Loss) = price where trade auto-exits to limit your loss.
                            TP (Take Profit) = price target where the trade closes in profit.
                        </div>
                        <div className="risk-row">
                            <div className="risk-item">
                                <div className="risk-label">Take Profit</div>
                                <div className="risk-value green">{rp.take_profit?.toFixed(4) || '—'}</div>
                                {tpPips && <div className="risk-pips">{tpPips} pips away</div>}
                            </div>
                            <div className="risk-item">
                                <div className="risk-label">Stop Loss</div>
                                <div className="risk-value red">{rp.stop_loss?.toFixed(4) || '—'}</div>
                                {slPips && <div className="risk-pips">{slPips} pips away</div>}
                            </div>
                            <div className="risk-item">
                                <div className="risk-label">Risk / Reward</div>
                                <div className="risk-value">{rp.risk_reward_ratio?.toFixed(2) || '—'}</div>
                                <div className="risk-pips">ratio</div>
                            </div>
                        </div>
                    </div>
                );
            })()}

            {sivIssues.length > 0 && <SIVAuditRow issues={sivIssues} />}
        </div>
    );
}

// ─── Main Component ───────────────────────────────────────────────────────────
export default function TradingAssistant({ analysisResult, loading, pair, chatHistory, chatRef, experienceLevel = 'beginner' }) {
    const [chatInput, setChatInput] = useState('');
    const [chatMessages, setChatMessages] = useState([]);
    const [chatLoading, setChatLoading] = useState(false);
    const [showHistory, setShowHistory] = useState(false);
    const [viewingSession, setViewingSession] = useState(null);

    // ── Simulation state ──────────────────────────────────────────────────────
    const [simLoading, setSimLoading] = useState(false);
    const [simResult, setSimResult] = useState(null);
    const [simAction, setSimAction] = useState(null); // 'BUY' | 'SELL' | 'HOLD' | null

    const messagesEndRef = useRef(null);
    const prevResultRef = useRef(null);

    useEffect(() => {
        if (chatRef) {
            chatRef.current = {
                messages: chatMessages,
                pair: analysisResult?.currency_pair || pair,
                result: analysisResult || null,
            };
        }
    }, [chatMessages, analysisResult, pair, chatRef]);

    // Reset chat + simulation when a new analysis arrives
    useEffect(() => {
        if (analysisResult && analysisResult !== prevResultRef.current) {
            setChatMessages([]);
            setSimResult(null);
            setSimAction(null);
            prevResultRef.current = analysisResult;
        }
    }, [analysisResult]);

    useEffect(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [analysisResult, loading, chatMessages, chatLoading, simResult, simLoading]);

    // ── Handle trade action ───────────────────────────────────────────────────
    const handleTradeAction = async (action) => {
        if (!analysisResult || simLoading) return;
        setSimAction(action);

        if (action === 'HOLD') {
            const holdMsg = {
                beginner: "No trade placed — you chose to wait. In forex, sitting out is a completely valid call. The signals are still worth studying even when no trade is taken.",
                basic: "You held — no position opened. Sometimes the best trade is no trade at all.",
                intermediate: "Standing aside. No position opened.",
            }[experienceLevel] ?? "No trade was placed. You chose to wait for a better opportunity.";

            setSimResult({ outcome: 'HOLD', action: 'HOLD', message: holdMsg, candles: [] });
            return;
        }

        const trade = analysisResult.trade || {};
        const entryPrice = analysisResult.tts?.price || 0;
        const slDistance = trade.sl_distance;
        const tpDistance = trade.tp_distance;

        if (!entryPrice || !slDistance || !tpDistance) {
            setSimResult({
                outcome: 'ERROR',
                message: 'Cannot simulate — entry price, SL, or TP is missing from this analysis.',
            });
            return;
        }

        setSimLoading(true);
        try {
            const result = await simulateTrade({
                currencyPair: analysisResult.currency_pair || pair,
                action,
                entryPrice,
                slDistance,
                tpDistance,
                targetDate: analysisResult.target_date,
            });
            setSimResult(result);
        } catch (err) {
            console.error('Simulation error:', err);
            setSimResult({
                outcome: 'ERROR',
                message: err?.response?.data?.error || 'Unable to reach the simulation endpoint. Make sure the backend is running.',
            });
        } finally {
            setSimLoading(false);
        }
    };

    // ── Chat ──────────────────────────────────────────────────────────────────
    const handleSend = async () => {
        if (!chatInput.trim() || chatLoading) return;
        const userMessage = chatInput.trim();
        setChatInput('');
        const newUserMsg = { role: 'user', content: userMessage };
        setChatMessages((prev) => [...prev, newUserMsg]);
        setChatLoading(true);
        try {
            const history = [...chatMessages, newUserMsg].map((m) => ({ role: m.role, content: m.content }));
            const response = await sendChatMessage(
                userMessage, pair, analysisResult?.analysis_id || null, history, experienceLevel,
                simResult || null,
            );
            setChatMessages((prev) => [
                ...prev,
                { role: 'assistant', content: response?.response || 'Sorry, I encountered an error. Please try again.' },
            ]);
        } catch (err) {
            console.error('Chat error:', err);
            setChatMessages((prev) => [
                ...prev,
                { role: 'assistant', content: 'Unable to reach the server. Make sure the Flask backend is running.' },
            ]);
        } finally {
            setChatLoading(false);
        }
    };

    const handleKeyDown = (e) => {
        if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleSend(); }
    };

    // ── Helpers ───────────────────────────────────────────────────────────────
    const hasEntryData = analysisResult
        && (analysisResult.trade?.sl_distance != null)
        && (analysisResult.tts?.price != null);

    return (
        <div className="card assistant-card">
            {/* Header */}
            <div className="card-header">
                <div className="card-header-icon" style={{ lineHeight: 0 }}>
                    <TradingTurtleMascot speaking={false} size={32} />
                </div>
                <div className="assistant-header-main">
                    <h2>Trading Assistant</h2>
                    <p>AI-Powered Analysis</p>
                </div>
                <button
                    className={`history-toggle-btn ${chatHistory.length === 0 ? 'disabled' : ''}`}
                    onClick={() => { if (chatHistory.length > 0) { setShowHistory(true); setViewingSession(null); } }}
                    title={chatHistory.length === 0 ? 'No past sessions yet' : 'View past sessions'}
                >
                    <History size={16} />
                    {chatHistory.length > 0 && (
                        <span className="history-badge">{chatHistory.length}</span>
                    )}
                </button>
            </div>

            {/* History panel */}
            {showHistory && (
                <div className="history-panel">
                    <div className="history-panel-header">
                        {viewingSession ? (
                            <>
                                <button className="history-back-btn" onClick={() => setViewingSession(null)}>
                                    <ChevronRight size={14} className="history-back-icon" />
                                    Back
                                </button>
                                <span className="history-panel-title">{viewingSession.pair}</span>
                                <span className="history-panel-time">{viewingSession.timestamp}</span>
                            </>
                        ) : (
                            <>
                                <span className="history-panel-title">Past Sessions</span>
                                <span className="history-panel-count">{chatHistory.length} session{chatHistory.length !== 1 ? 's' : ''}</span>
                            </>
                        )}
                        <button className="history-close-btn" onClick={() => { setShowHistory(false); setViewingSession(null); }}>
                            <X size={14} />
                        </button>
                    </div>

                    <div className="history-panel-body">
                        {!viewingSession && (
                            chatHistory.length === 0 ? (
                                <p className="history-empty-note">No past sessions yet.</p>
                            ) : (
                                [...chatHistory].reverse().map((session) => {
                                    const decision = session.result?.verdict?.decision || null;
                                    return (
                                        <button
                                            key={session.id}
                                            className="history-session-item"
                                            onClick={() => setViewingSession(session)}
                                        >
                                            <div className="history-session-top">
                                                <span className="history-session-pair">{session.pair}</span>
                                                {decision && <SignalBadge signal={decision} />}
                                            </div>
                                            <div className="history-session-meta">
                                                <span>{session.timestamp}</span>
                                                <span>{session.messages.length} message{session.messages.length !== 1 ? 's' : ''}</span>
                                            </div>
                                            {session.messages.length > 0 && (
                                                <div className="history-session-preview">
                                                    {session.messages[session.messages.length - 1]?.content?.slice(0, 80)}...
                                                </div>
                                            )}
                                            <ChevronRight size={14} className="history-session-arrow" />
                                        </button>
                                    );
                                })
                            )
                        )}

                        {viewingSession && (
                            <div className="history-messages">
                                {viewingSession.result && (
                                    <>
                                        <div className="chat-msg chat-msg-assistant">
                                            <div className="chat-msg-icon" style={{ background: 'none', border: 'none', width: 32, height: 32 }}>
                                                <TradingTurtleMascot speaking={false} size={32} />
                                            </div>
                                            <div className="chat-msg-bubble chat-msg-bubble--analysis">
                                                <AnalysisCard result={viewingSession.result} pairLabel={viewingSession.pair} />
                                            </div>
                                        </div>
                                        {viewingSession.messages.length > 0 && (
                                            <div className="history-divider">follow-up chat</div>
                                        )}
                                    </>
                                )}

                                {viewingSession.messages.map((msg, idx) => (
                                    <div key={idx} className={`chat-msg chat-msg-${msg.role}`}>
                                        <div
                                            className="chat-msg-icon"
                                            style={msg.role === 'assistant'
                                                ? { background: 'none', border: 'none', width: 32, height: 32 }
                                                : {}}
                                        >
                                            {msg.role === 'user'
                                                ? <User size={14} />
                                                : <TradingTurtleMascot speaking={false} size={32} />
                                            }
                                        </div>
                                        <div className="chat-msg-bubble">
                                            {msg.role === 'assistant' ? formatBoldText(msg.content) : msg.content}
                                        </div>
                                    </div>
                                ))}

                                {viewingSession.messages.length === 0 && !viewingSession.result && (
                                    <p className="history-empty-note">No messages in this session.</p>
                                )}
                            </div>
                        )}
                    </div>
                </div>
            )}

            {/* Messages feed */}
            <div className="assistant-messages">

                {/* Main analysis loading */}
                {loading && (
                    <div style={{
                        display: 'flex', flexDirection: 'column', alignItems: 'center',
                        justifyContent: 'center', gap: 12,
                        padding: '32px 20px',
                        background: 'rgba(46,125,50,0.06)',
                        border: '1px solid rgba(46,168,74,0.18)',
                        borderRadius: 12, margin: '12px 0',
                    }}>
                        <TradingTurtleMascot thinking={true} size={80} />
                        <div style={{ textAlign: 'center' }}>
                            <p style={{ margin: '0 0 4px', fontSize: 14, fontWeight: 600, color: 'var(--text-primary, #e2e8f0)' }}>
                                Shelly is crunching the market data...
                            </p>
                            <p style={{ margin: 0, fontSize: 11, color: 'var(--text-muted, #64748b)' }}>
                                Running CE · TTS · SIV · Verdict agents — this may take a minute.
                            </p>
                        </div>
                        <div style={{ display: 'flex', gap: 8, alignItems: 'center', marginTop: 4 }}>
                            {['CE', 'TTS', 'SIV', 'Verdict'].map((label, i) => (
                                <React.Fragment key={label}>
                                    <span style={{
                                        fontSize: 10, fontWeight: 700, letterSpacing: '0.05em',
                                        color: 'var(--text-muted, #64748b)',
                                        padding: '3px 8px',
                                        border: '1px solid rgba(46,168,74,0.25)',
                                        borderRadius: 99,
                                        animation: `agentPulse 1.6s ${i * 0.3}s ease-in-out infinite`,
                                    }}>
                                        {label}
                                    </span>
                                    {i < 3 && <span style={{ fontSize: 10, color: 'rgba(46,168,74,0.4)' }}>→</span>}
                                </React.Fragment>
                            ))}
                        </div>
                        <style>{`
                            @keyframes agentPulse {
                                0%,100% { opacity: 0.35; border-color: rgba(46,168,74,0.2); }
                                50%      { opacity: 1;    border-color: rgba(46,168,74,0.8); color: #2EA84A; }
                            }
                        `}</style>
                    </div>
                )}

                {/* Empty state */}
                {!loading && !analysisResult && chatMessages.length === 0 && (
                    <div className="assistant-empty-state" style={{
                        display: 'flex', flexDirection: 'column', alignItems: 'center',
                        gap: 12, padding: '28px 16px',
                        color: 'var(--text-muted, #64748b)',
                    }}>
                        <TradingTurtleMascot speaking={false} size={88} />
                        <div style={{ textAlign: 'center' }}>
                            <p style={{ margin: '0 0 4px', fontSize: 14, fontWeight: 600, color: 'var(--text-secondary, #94a3b8)' }}>
                                Hi, I'm Shelly!
                            </p>
                            <p style={{ margin: 0, fontSize: 12, lineHeight: 1.6, color: 'var(--text-muted, #64748b)' }}>
                                Set up your parameters, pick a date on the chart, then click <strong style={{ color: '#2EA84A' }}>Review & Analyse.</strong>
                            </p>
                        </div>
                    </div>
                )}

                {/* Analysis result + action buttons + simulation */}
                {analysisResult && !loading && (
                    <>
                        <div className="chat-msg chat-msg-assistant">
                            <div className="chat-msg-icon" style={{ background: 'none', border: 'none', width: 32, height: 32 }}>
                                <TradingTurtleMascot speaking={false} size={32} />
                            </div>
                            <div className="chat-msg-bubble chat-msg-bubble--analysis">
                                <AnalysisCard result={analysisResult} pairLabel={pair} />

                                {/* ── Action buttons — only show if we have TP/SL data and no action taken yet ── */}
                                {!simAction && (
                                    <TradeActionButtons
                                        onAction={handleTradeAction}
                                        disabled={simLoading}
                                    />
                                )}

                                {/* ── Simulation loading ── */}
                                {simLoading && (
                                    <div style={{
                                        display: 'flex', alignItems: 'center', gap: 10,
                                        padding: '12px 14px', marginTop: 10,
                                        background: 'rgba(46,168,74,0.06)',
                                        border: '1px solid rgba(46,168,74,0.2)',
                                        borderRadius: 'var(--radius-sm)',
                                        fontSize: 12, color: 'var(--text-muted)',
                                    }}>
                                        <TradingTurtleMascot thinking={true} size={28} />
                                        <div>
                                            <div style={{ fontWeight: 600, color: 'var(--text-secondary)', marginBottom: 2 }}>
                                                Simulating {simAction} over 5 candles...
                                            </div>
                                            <div style={{ fontSize: 11 }}>
                                                Fetching OHLCV data and checking TP / SL levels
                                            </div>
                                        </div>
                                    </div>
                                )}

                                {/* ── Simulation result ── */}
                                {simResult && !simLoading && (
                                    <>
                                        {simResult.outcome === 'ERROR' ? (
                                            <div style={{
                                                margin: '12px 15px', padding: '12px 14px',
                                                borderRadius: 'var(--radius-sm)',
                                                border: '1px solid rgba(220,53,69,0.3)',
                                                background: 'rgba(220,53,69,0.06)',
                                                fontSize: 12, color: '#dc3545',
                                            }}>
                                                <AlertTriangle size={13} style={{ marginRight: 6 }} />
                                                {simResult.message}
                                            </div>
                                        ) : (
                                            <SimTradeResult simResult={simResult} analysisResult={analysisResult} experienceLevel={experienceLevel} />
                                        )}

                                        {/* Reset button to try a different action */}
                                        {simResult.outcome !== 'ERROR' && (
                                            <button
                                                onClick={() => { setSimResult(null); setSimAction(null); }}
                                                style={{
                                                    marginTop: 8,
                                                    padding: '5px 12px',
                                                    borderRadius: 6,
                                                    border: '1px solid var(--bg-input-border)',
                                                    background: 'transparent',
                                                    color: 'var(--text-muted)',
                                                    fontSize: 11, cursor: 'pointer',
                                                    display: 'flex', alignItems: 'center', gap: 5,
                                                }}
                                            >
                                                <RefreshCw size={11} /> Try a different action
                                            </button>
                                        )}
                                    </>
                                )}
                            </div>
                        </div>
                    </>
                )}

                {/* Chat messages */}
                {chatMessages.map((msg, idx) => (
                    <div key={idx} className={`chat-msg chat-msg-${msg.role}`}>
                        <div className="chat-msg-icon" style={msg.role === 'assistant'
                            ? { background: 'none', border: 'none', width: 32, height: 32 }
                            : {}
                        }>
                            {msg.role === 'user'
                                ? <User size={14} />
                                : <TradingTurtleMascot speaking={false} size={32} />
                            }
                        </div>
                        <div className="chat-msg-bubble">
                            {msg.role === 'assistant' ? formatBoldText(msg.content) : msg.content}
                        </div>
                    </div>
                ))}

                {/* Chat follow-up loading */}
                {chatLoading && (
                    <div className="chat-msg chat-msg-assistant">
                        <div className="chat-msg-icon" style={{ lineHeight: 0 }}>
                            <TradingTurtleMascot thinking={true} size={28} />
                        </div>
                        <div className="chat-msg-bubble" style={{
                            display: 'flex', alignItems: 'center', gap: 8, padding: '8px 12px',
                        }}>
                            <span style={{ fontSize: 12, color: 'var(--text-muted, #64748b)' }}>
                                Shelly is thinking
                            </span>
                            <span style={{ display: 'flex', gap: 3, alignItems: 'center' }}>
                                {[0, 0.2, 0.4].map((delay, i) => (
                                    <span key={i} style={{
                                        width: 4, height: 4, borderRadius: '50%',
                                        background: '#2EA84A', display: 'inline-block',
                                        animation: `chatDot 1s ${delay}s ease-in-out infinite`,
                                    }} />
                                ))}
                            </span>
                            <style>{`
                                @keyframes chatDot {
                                    0%,100% { opacity: 0.2; transform: translateY(0); }
                                    50%     { opacity: 1;   transform: translateY(-3px); }
                                }
                            `}</style>
                        </div>
                    </div>
                )}

                <div ref={messagesEndRef} />
            </div>

            {/* Input row */}
            <div className="chat-input-row">
                <input
                    type="text"
                    className="chat-input"
                    value={chatInput}
                    onChange={(e) => setChatInput(e.target.value)}
                    onKeyDown={handleKeyDown}
                    disabled={loading || chatLoading || simLoading}
                    placeholder={
                        loading ? 'Waiting for Shelly...'
                            : simLoading ? 'Simulating trade...'
                                : chatLoading ? 'Shelly is thinking...'
                                    : analysisResult ? 'Ask a follow-up question...'
                                        : 'Ask about the analysis...'
                    }
                />
                <button
                    className="chat-send-btn"
                    onClick={handleSend}
                    disabled={!chatInput.trim() || loading || chatLoading || simLoading}
                >
                    <Send size={16} />
                </button>
            </div>
        </div>
    );
}