import React, { useState, useRef, useEffect } from 'react';
import { Activity, BookOpen, Sun, Moon, ShieldAlert, X, ChevronLeft, ChevronRight } from 'lucide-react';
import TradingParameters from './components/TradingParameters';
import CandlestickChart from './components/CandlestickChart';
import TradingAssistant from './components/TradingAssistant';
import Backtesting from './components/Backtesting';
import { runOrchestrator } from './services/api';
import TradingTurtleMascot from './components/TradingTurtleMascot';
import tutorialStep1 from '../videos/1.png';
import tutorialStep2 from '../videos/2.png';
import tutorialStep3 from '../videos/3.png';
import './styles/backtesting.css';


const ENABLE_LLM_ANALYSIS = true;
const THEME_STORAGE_KEY = 'trading_assistant_theme';
const DEFAULT_NARRATION_DURATION_MS = 3500;

const TRADING_TUTORIAL_SLIDES = [
    {
        id: 'trading-select',
        image: tutorialStep1,
        alt: 'Step one showing trade input setup fields.',
        title: 'Understand trading parameters',
        narration: 'Simple explanation: Forex Pair is the market you want to analyze. Account Capital is your money size. Leverage controls how big your exposure is. Risk Threshold controls how strict the system is with risky setups. Analogy: This is like planning a trip, pair is your route, capital is your budget, leverage is your speed, and risk threshold is your safety rule.'
    },
    {
        id: 'trading-process',
        image: tutorialStep2,
        alt: 'Step two showing multi-agent analysis flow.',
        title: 'What happens inside the system',
        narration: 'Simple explanation: After you click Run Analysis, the system checks price data, news sentiment, and risk rules, then combines those checks into one recommendation. Analogy: It is like asking three helpers before a trip, one checks weather, one checks traffic, and one checks safety, then they give one final go or stop suggestion.'
    },
    {
        id: 'trading-act',
        image: tutorialStep3,
        alt: 'Step three showing recommendation and risk summary panels.',
        title: 'Read output like a beginner',
        narration: 'Simple explanation: Buy means conditions look favorable, Sell means the opposite direction looks stronger, and Hold means no clear setup yet. Always read the reason and risk notes before acting. Analogy: Treat it like a traffic light, buy is green, sell is red, and hold is yellow, slow down and wait for clearer conditions.'
    }
];

const BACKTESTING_TUTORIAL_SLIDES = [
    {
        id: 'backtesting-select',
        image: tutorialStep1,
        alt: 'Step one showing backtesting setup and pair selection.',
        title: 'Choose pair and day',
        narration: 'Simple explanation: In backtesting, you choose a pair and a past date to test how the strategy would have behaved. No live money is involved. Analogy: It is like watching a game replay to study decisions before playing a real match.'
    },
    {
        id: 'backtesting-process',
        image: tutorialStep2,
        alt: 'Step two showing historical simulation and agent scoring.',
        title: 'Inspect simulation mechanics',
        narration: 'Simple explanation: The system runs the same analysis logic on historical data and generates signals for that day. This helps you see if results stay stable in different conditions. Analogy: It is like a driving simulator where you practice in rain, traffic, and clear roads before driving outside.'
    },
    {
        id: 'backtesting-act',
        image: tutorialStep3,
        alt: 'Step three showing performance and decision summary.',
        title: 'Compare outcomes and refine',
        narration: 'Simple explanation: Review profit and loss, wins versus losses, and how reliable signals were. Use that feedback to improve rules before going live. Analogy: It is like taking mock exams first, then fixing weak topics before the real test.'
    }
];

export default function App() {
    const [targetDate, setTargetDate] = useState(null);
    const [currentView, setCurrentView] = useState(() => {
        const hash = window.location.hash.slice(1).replace('/', '') || '';
        return hash === 'backtesting' ? 'backtesting' : 'trading';
    });

    const [pair, setPair] = useState('EUR/USD');
    const [amount, setAmount] = useState('1000');
    const [leverage, setLeverage] = useState('1:1');
    const [riskThreshold, setRiskThreshold] = useState('1');

    const [loading, setLoading] = useState(false);
    const [analysisResult, setAnalysisResult] = useState(null);
    const [error, setError] = useState(null);

    const [chatHistory, setChatHistory] = useState([]);
    const [showTutorial, setShowTutorial] = useState(true);
    const [showIntro, setShowIntro] = useState(true);
    const [theme, setTheme] = useState('dark');
    const [tutorialSlideIndex, setTutorialSlideIndex] = useState(0);
    const [typedNarration, setTypedNarration] = useState('');
    const [isNarrating, setIsNarrating] = useState(false);
    const [speakingDurationMs, setSpeakingDurationMs] = useState(DEFAULT_NARRATION_DURATION_MS);

    const [experienceLevel, setExperienceLevel] = useState(null);
    const [introPhase, setIntroPhase] = useState('welcome');

    const chatRef = useRef({ messages: [], pair: null, result: null });
    const tutorialSlides = currentView === 'backtesting' ? BACKTESTING_TUTORIAL_SLIDES : TRADING_TUTORIAL_SLIDES;
    const activeTutorialSlide = tutorialSlides[tutorialSlideIndex] || tutorialSlides[0];

    useEffect(() => {
        const handleHashChange = () => {
            const hash = window.location.hash.slice(1).replace('/', '') || '';
            setCurrentView(hash === 'backtesting' ? 'backtesting' : 'trading');
        };
        window.addEventListener('hashchange', handleHashChange);
        return () => window.removeEventListener('hashchange', handleHashChange);
    }, []);

    useEffect(() => {
        const savedTheme = window.localStorage.getItem(THEME_STORAGE_KEY);
        if (savedTheme) setTheme(savedTheme);
    }, []);

    useEffect(() => {
        document.documentElement.setAttribute('data-theme', theme);
        window.localStorage.setItem(THEME_STORAGE_KEY, theme);
    }, [theme]);

    const closeTutorial = () => setShowTutorial(false);
    const openTutorial = () => {
        setTutorialSlideIndex(0);
        setShowIntro(true);
        setIntroPhase('welcome');
        setShowTutorial(true);
    };
    const toggleTheme = () => setTheme((prev) => (prev === 'dark' ? 'light' : 'dark'));

    const goToPreviousTutorialSlide = () =>
        setTutorialSlideIndex((prev) => (prev - 1 + tutorialSlides.length) % tutorialSlides.length);
    const goToNextTutorialSlide = () =>
        setTutorialSlideIndex((prev) => (prev + 1) % tutorialSlides.length);

    useEffect(() => {
        if (!showTutorial) return;
        const handleEsc = (e) => e.key === 'Escape' && closeTutorial();
        window.addEventListener('keydown', handleEsc);
        return () => window.removeEventListener('keydown', handleEsc);
    }, [showTutorial]);

    useEffect(() => { setTutorialSlideIndex(0); }, [currentView]);

    useEffect(() => {
        if (!showTutorial || !activeTutorialSlide) {
            setTypedNarration('');
            setIsNarrating(false);
            return;
        }

        const fullText = activeTutorialSlide.narration;
        const duration = Math.max(300, speakingDurationMs);
        let animationFrameId = 0;
        const startTime = performance.now();

        setTypedNarration('');
        setIsNarrating(true);

        const animateTyping = (now) => {
            const progress = Math.min(1, (now - startTime) / duration);
            const charCount = Math.floor(progress * fullText.length);
            setTypedNarration(fullText.slice(0, charCount));
            if (progress >= 1) {
                setTypedNarration(fullText);
                setIsNarrating(false);
                return;
            }
            animationFrameId = window.requestAnimationFrame(animateTyping);
        };

        animationFrameId = window.requestAnimationFrame(animateTyping);
        return () => window.cancelAnimationFrame(animationFrameId);
    }, [showTutorial, tutorialSlideIndex, activeTutorialSlide, speakingDurationMs]);

    const handleGetSentiment = async () => {
        if (chatRef.current.messages.length > 0 || chatRef.current.result) {
            setChatHistory((prev) => [
                ...prev,
                {
                    id: Date.now(),
                    pair: chatRef.current.pair || pair,
                    messages: [...chatRef.current.messages],
                    result: chatRef.current.result,
                    timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
                },
            ]);
        }

        setLoading(true);
        setError(null);

        try {
            const result = await runOrchestrator(pair, !ENABLE_LLM_ANALYSIS, targetDate, {
                accountCapital: amount,
                leverage,
                riskThreshold,
                experienceLevel,
            });
            setAnalysisResult(result);
        } catch (err) {
            console.error('Orchestrator error:', err);
            setError(err.response?.data?.error || err.message || 'Analysis failed');
            setAnalysisResult(null);
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="app">
            {/* ─── Header ─────────────────────────────────────────────────── */}
            <header className="app-header">
                <div className="app-header-brand">
                    <div className="app-header-icon" style={{ lineHeight: 0 }}>
                        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32"
                            shapeRendering="crispEdges" width="36" height="36"
                            style={{ imageRendering: 'pixelated', display: 'block' }}
                            aria-hidden="true">
                            <style>{`
                                @keyframes hdrFloat { 0%,100%{transform:translateY(0)} 50%{transform:translateY(-1px)} }
                                .hdr-body { animation: hdrFloat 2.8s ease-in-out infinite; transform-origin: 16px 28px; }
                            `}</style>
                            <g className="hdr-body">
                                {/* shell */}
                                <rect x="5" y="6" width="22" height="16" fill="#2E7D32" />
                                <rect x="4" y="21" width="24" height="2" fill="#1B5E20" />
                                {/* shell shine */}
                                <rect x="8" y="8" width="6" height="4" fill="#3A9941" opacity="0.6" />
                                <rect x="16" y="8" width="6" height="4" fill="#3A9941" opacity="0.6" />
                                {/* head */}
                                <rect x="9" y="9" width="14" height="5" fill="#81C784" />
                                {/* eyes */}
                                <rect x="11" y="10" width="2" height="3" fill="#1A1A1A" />
                                <rect x="19" y="10" width="2" height="3" fill="#1A1A1A" />
                                {/* mouth */}
                                <rect x="7" y="14" width="18" height="2" fill="#1A1A1A" />
                                {/* legs */}
                                <rect x="9" y="23" width="4" height="3" fill="#81C784" />
                                <rect x="19" y="23" width="4" height="3" fill="#81C784" />
                                {/* arms */}
                                <rect x="3" y="17" width="3" height="3" fill="#81C784" />
                                <rect x="26" y="17" width="3" height="3" fill="#81C784" />
                                {/* coin */}
                                <rect x="27" y="21" width="4" height="4" fill="#FBC02D" />
                                <rect x="28" y="20" width="2" height="1" fill="#FBC02D" />
                                <rect x="28" y="25" width="2" height="1" fill="#FBC02D" />
                            </g>
                        </svg>
                    </div>
                    <div>
                        <h1>{currentView === 'backtesting' ? 'Backtesting' : 'Trading Assistant'}</h1>
                        <p>{currentView === 'backtesting' ? 'Historical Strategy Performance' : 'AI-Powered Forex Analysis'}</p>
                    </div>
                </div>
                <div className="app-header-actions">
                    <button className="header-action-btn" onClick={openTutorial}>
                        <BookOpen size={15} />
                        Inside the System
                    </button>
                    <button className="header-action-btn header-theme-btn" onClick={toggleTheme}>
                        {theme === 'dark' ? <Sun size={15} /> : <Moon size={15} />}
                        {theme === 'dark' ? 'Light Mode' : 'Dark Mode'}
                    </button>
                </div>
            </header>

            {/* ─── Body ───────────────────────────────────────────────────── */}
            <main className={`app-body app-body-${currentView}`}>
                {currentView === 'backtesting' ? (
                    <Backtesting />
                ) : (
                    <>
                        <TradingParameters
                            pair={pair} setPair={setPair}
                            amount={amount} setAmount={setAmount}
                            leverage={leverage} setLeverage={setLeverage}
                            riskThreshold={riskThreshold} setRiskThreshold={setRiskThreshold}
                            onSubmit={handleGetSentiment}
                            loading={loading}
                        />
                        <CandlestickChart pair={pair} ohlcvData={null} theme={theme} onDateChange={setTargetDate} />
                        <TradingAssistant
                            analysisResult={analysisResult}
                            loading={loading}
                            pair={pair}
                            chatHistory={chatHistory}
                            chatRef={chatRef}
                            experienceLevel={experienceLevel}
                        />
                    </>
                )}
            </main>

            {/* ─── Error Toast ────────────────────────────────────────────── */}
            {error && (
                <div className="error-toast" style={{
                    position: 'fixed', bottom: 24, left: '50%', transform: 'translateX(-50%)',
                    background: 'var(--danger-toast-bg)', color: 'var(--on-danger)',
                    padding: '12px 24px', borderRadius: 'var(--radius-md)', fontSize: 13,
                    fontWeight: 500, zIndex: 150, backdropFilter: 'blur(8px)',
                    display: 'flex', alignItems: 'center', gap: '12px'
                }}>
                    <ShieldAlert size={18} />
                    {error}
                    <button onClick={() => setError(null)} style={{ background: 'none', border: 'none', color: 'inherit', cursor: 'pointer', padding: '4px' }}>
                        <X size={18} />
                    </button>
                </div>
            )}

            {/* ─── Tutorial Modal ─────────────────────────────────────────── */}
            {showTutorial && (
                <div
                    className="tutorial-overlay"
                    onClick={closeTutorial}
                    style={{
                        position: 'fixed', inset: 0, zIndex: 200,
                        background: 'rgba(0,0,0,0.65)',
                        backdropFilter: 'blur(6px)',
                        display: 'flex', alignItems: 'center', justifyContent: 'center',
                        padding: '16px',
                    }}
                >
                    <div
                        className="tutorial-modal"
                        onClick={(e) => e.stopPropagation()}
                        style={{
                            width: '100%', maxWidth: 620,
                            background: 'var(--bg-card, #111)',
                            border: '1px solid rgba(46,168,74,0.35)',
                            borderRadius: 16,
                            boxShadow: '0 0 0 1px rgba(46,168,74,0.12), 0 24px 64px rgba(0,0,0,0.6)',
                            overflow: 'hidden',
                            display: 'flex', flexDirection: 'column',
                        }}
                    >
                        {/* ══════════════════════════════════════════
                INTRO SCREEN
            ══════════════════════════════════════════ */}
                        {showIntro ? (
                            <>
                                {/* ── Close button ── */}
                                <div style={{ display: 'flex', justifyContent: 'flex-end', padding: '12px 16px 0' }}>
                                    <button onClick={closeTutorial} style={{
                                        background: 'none', border: 'none', cursor: 'pointer',
                                        color: 'var(--text-muted, #888)', fontSize: 20,
                                        lineHeight: 1, padding: '4px 8px', borderRadius: 6,
                                    }} aria-label="Close">×</button>
                                </div>

                                {introPhase === 'welcome' ? (
                                    /* ══ PHASE 1 — Welcome ══════════════════════════════════════ */
                                    <>
                                        <div style={{
                                            display: 'flex', flexDirection: 'column', alignItems: 'center',
                                            padding: '8px 32px 24px', gap: 16, textAlign: 'center',
                                        }}>
                                            <TradingTurtleMascot speaking={true} size={120} />

                                            <div>
                                                <h2 style={{
                                                    margin: '0 0 8px', fontSize: 22, fontWeight: 700,
                                                    color: 'var(--text-primary, #fff)', lineHeight: 1.25,
                                                }}>
                                                    Hi, I'm Shelly! 👋
                                                </h2>
                                                <p style={{
                                                    margin: '0 0 6px', fontSize: 13, fontWeight: 600,
                                                    color: '#2EA84A', letterSpacing: '0.04em', textTransform: 'uppercase',
                                                }}>
                                                    Your Forex Learning Companion
                                                </p>
                                            </div>

                                            <div style={{
                                                background: 'rgba(46,125,50,0.08)',
                                                border: '1px solid rgba(46,168,74,0.2)',
                                                borderRadius: 12, padding: '16px 20px',
                                                fontSize: 13, lineHeight: 1.75,
                                                color: 'var(--text-secondary, #94a3b8)',
                                                textAlign: 'left', maxWidth: 480,
                                            }}>
                                                <p style={{ margin: '0 0 10px' }}>
                                                    This is the <strong style={{ color: 'var(--text-primary, #fff)' }}>
                                                        Multi-Agent Trading System (MAS)</strong> — a tool designed
                                                    to help <strong style={{ color: '#2EA84A' }}>beginners</strong> explore
                                                    the forex market without feeling lost.
                                                </p>
                                                <p style={{ margin: '0 0 10px' }}>
                                                    It runs <strong style={{ color: 'var(--text-primary, #fff)' }}>four AI agents</strong> behind
                                                    the scenes — one reads charts, one scans the news, one checks if they agree,
                                                    and one makes the final call — then explains everything in plain language.
                                                </p>
                                                <p style={{ margin: 0 }}>
                                                    I'll walk you through how it works, step by step. No trading experience needed. 🐢
                                                </p>
                                            </div>

                                            <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', justifyContent: 'center' }}>
                                                {['📊 Technical Analysis', '📰 News Sentiment', '✅ Signal Integrity', '🎯 Risk Sizing'].map((pill) => (
                                                    <span key={pill} style={{
                                                        fontSize: 11, fontWeight: 600,
                                                        padding: '4px 12px', borderRadius: 99,
                                                        background: 'rgba(46,168,74,0.12)',
                                                        border: '1px solid rgba(46,168,74,0.25)',
                                                        color: '#2EA84A',
                                                    }}>{pill}</span>
                                                ))}
                                            </div>
                                        </div>

                                        <div style={{
                                            padding: '14px 20px 20px', display: 'flex',
                                            justifyContent: 'center', gap: 12,
                                            borderTop: '1px solid rgba(46,168,74,0.12)',
                                        }}>
                                            <button onClick={closeTutorial} style={{
                                                background: 'none', border: '1px solid rgba(46,168,74,0.3)',
                                                borderRadius: 8, padding: '9px 20px', fontSize: 13,
                                                color: 'var(--text-secondary, #94a3b8)', cursor: 'pointer',
                                            }}>
                                                Skip for now
                                            </button>
                                            <button onClick={() => setIntroPhase('question')} style={{
                                                background: '#2EA84A', color: '#fff', border: 'none',
                                                borderRadius: 8, padding: '9px 28px', fontSize: 14,
                                                fontWeight: 600, cursor: 'pointer', letterSpacing: '0.02em',
                                            }}>
                                                Let's get started →
                                            </button>
                                        </div>
                                    </>
                                ) : (
                                    /* ══ PHASE 2 — Experience Question ══════════════════════════ */
                                    <>
                                        <div style={{
                                            display: 'flex', flexDirection: 'column', alignItems: 'center',
                                            padding: '16px 28px 24px', gap: 20, textAlign: 'center',
                                        }}>
                                            <TradingTurtleMascot speaking={false} size={80} />

                                            <div>
                                                <h2 style={{
                                                    margin: '0 0 6px', fontSize: 18, fontWeight: 700,
                                                    color: 'var(--text-primary, #fff)',
                                                }}>
                                                    Before we dive in — quick question!
                                                </h2>
                                                <p style={{
                                                    margin: 0, fontSize: 13,
                                                    color: 'var(--text-secondary, #94a3b8)',
                                                }}>
                                                    How familiar are you with forex trading?
                                                    <br />
                                                    <span style={{ fontSize: 11, color: 'var(--text-muted, #64748b)' }}>
                                                        I'll tailor my explanations to match your level.
                                                    </span>
                                                </p>
                                            </div>

                                            {/* Option cards */}
                                            <div style={{ display: 'flex', flexDirection: 'column', gap: 10, width: '100%', maxWidth: 440 }}>
                                                {[
                                                    {
                                                        level: 'beginner',
                                                        emoji: '🌱',
                                                        label: "I'm completely new",
                                                        sub: "Never traded before — start from scratch please",
                                                    },
                                                    {
                                                        level: 'basic',
                                                        emoji: '📈',
                                                        label: 'I know the basics',
                                                        sub: "I've heard of pips, charts, and maybe a few indicators",
                                                    },
                                                    {
                                                        level: 'intermediate',
                                                        emoji: '⚙️',
                                                        label: 'I have some experience',
                                                        sub: "I trade or have traded — just learning this system",
                                                    },
                                                ].map(({ level, emoji, label, sub }) => (
                                                    <button
                                                        key={level}
                                                        onClick={() => {
                                                            setExperienceLevel(level);
                                                            setShowIntro(false);
                                                        }}
                                                        style={{
                                                            display: 'flex', alignItems: 'center', gap: 14,
                                                            padding: '14px 16px', borderRadius: 10, cursor: 'pointer',
                                                            textAlign: 'left', width: '100%',
                                                            background: experienceLevel === level
                                                                ? 'rgba(46,168,74,0.15)'
                                                                : 'rgba(46,125,50,0.06)',
                                                            border: experienceLevel === level
                                                                ? '1px solid rgba(46,168,74,0.6)'
                                                                : '1px solid rgba(46,168,74,0.18)',
                                                            transition: 'all 0.15s ease',
                                                        }}
                                                        onMouseEnter={e => {
                                                            e.currentTarget.style.background = 'rgba(46,168,74,0.12)';
                                                            e.currentTarget.style.borderColor = 'rgba(46,168,74,0.45)';
                                                        }}
                                                        onMouseLeave={e => {
                                                            e.currentTarget.style.background = experienceLevel === level
                                                                ? 'rgba(46,168,74,0.15)' : 'rgba(46,125,50,0.06)';
                                                            e.currentTarget.style.borderColor = experienceLevel === level
                                                                ? 'rgba(46,168,74,0.6)' : 'rgba(46,168,74,0.18)';
                                                        }}
                                                    >
                                                        <span style={{ fontSize: 24, flexShrink: 0 }}>{emoji}</span>
                                                        <div>
                                                            <p style={{
                                                                margin: '0 0 2px', fontSize: 13, fontWeight: 600,
                                                                color: 'var(--text-primary, #fff)',
                                                            }}>{label}</p>
                                                            <p style={{
                                                                margin: 0, fontSize: 11,
                                                                color: 'var(--text-secondary, #94a3b8)',
                                                            }}>{sub}</p>
                                                        </div>
                                                    </button>
                                                ))}
                                            </div>
                                        </div>

                                        <div style={{
                                            padding: '12px 20px 18px', display: 'flex',
                                            justifyContent: 'space-between', alignItems: 'center',
                                            borderTop: '1px solid rgba(46,168,74,0.12)',
                                        }}>
                                            <button onClick={() => setIntroPhase('welcome')} style={{
                                                background: 'none', border: 'none', cursor: 'pointer',
                                                fontSize: 12, color: 'var(--text-muted, #64748b)', padding: '4px 0',
                                            }}>
                                                ← Back
                                            </button>
                                            <button
                                                onClick={() => setShowIntro(false)}
                                                style={{
                                                    background: 'rgba(46,168,74,0.15)',
                                                    border: '1px solid rgba(46,168,74,0.3)',
                                                    borderRadius: 8, padding: '8px 18px', fontSize: 12,
                                                    color: 'var(--text-secondary, #94a3b8)', cursor: 'pointer',
                                                }}
                                            >
                                                Skip question
                                            </button>
                                        </div>
                                    </>
                                )}
                            </>
                        ) : (
                            /* ══════════════════════════════════════════
                               TUTORIAL SLIDES (existing content)
                            ══════════════════════════════════════════ */
                            <>
                                {/* Header bar */}
                                <div style={{
                                    display: 'flex', alignItems: 'center', justifyContent: 'space-between',
                                    padding: '14px 20px',
                                    borderBottom: '1px solid rgba(46,168,74,0.18)',
                                    background: 'rgba(46,125,50,0.12)',
                                }}>
                                    <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                                        <TradingTurtleMascot speaking={false} size={28} />
                                        <div>
                                            <p style={{ margin: 0, fontSize: 15, fontWeight: 600, color: 'var(--text-primary, #fff)' }}>
                                                {currentView === 'backtesting' ? 'Backtesting Guide' : 'Inside the Trading System'}
                                            </p>
                                            <p style={{ margin: 0, fontSize: 12, color: 'var(--text-secondary, #8a9)', lineHeight: 1.3 }}>
                                                {currentView === 'backtesting'
                                                    ? 'Historical strategy performance'
                                                    : 'Beginner-friendly · 3 steps'}
                                            </p>
                                        </div>
                                    </div>
                                    <button
                                        onClick={closeTutorial}
                                        style={{
                                            background: 'none', border: 'none', cursor: 'pointer',
                                            color: 'var(--text-muted, #888)',
                                            fontSize: 20, lineHeight: 1, padding: '4px 8px', borderRadius: 6,
                                        }}
                                        aria-label="Close tutorial"
                                    >×</button>
                                </div>

                                {/* Slide image */}
                                <div style={{ position: 'relative', background: 'rgba(0,0,0,0.25)' }}>
                                    <img
                                        src={activeTutorialSlide.image}
                                        alt={activeTutorialSlide.alt}
                                        style={{ width: '100%', display: 'block', maxHeight: 260, objectFit: 'contain' }}
                                    />
                                    <span style={{
                                        position: 'absolute', top: 10, left: 12,
                                        background: 'rgba(46,168,74,0.85)',
                                        color: '#fff', fontSize: 11, fontWeight: 700,
                                        padding: '3px 10px', borderRadius: 99, letterSpacing: '0.06em',
                                    }}>
                                        STEP {tutorialSlideIndex + 1} / {tutorialSlides.length}
                                    </span>
                                </div>

                                {/* Nav + dots */}
                                <div style={{
                                    display: 'flex', alignItems: 'center', justifyContent: 'center',
                                    gap: 14, padding: '12px 20px 4px',
                                }}>
                                    <button onClick={goToPreviousTutorialSlide} style={{
                                        background: 'none', border: '1px solid rgba(46,168,74,0.35)',
                                        borderRadius: 8, cursor: 'pointer', padding: '6px 10px',
                                        color: 'var(--text-secondary, #8a9)', display: 'flex', alignItems: 'center',
                                    }} aria-label="Previous step">
                                        <ChevronLeft size={16} />
                                    </button>

                                    <div style={{ display: 'flex', gap: 6 }}>
                                        {tutorialSlides.map((slide, idx) => (
                                            <button key={slide.id} onClick={() => setTutorialSlideIndex(idx)}
                                                aria-label={`Step ${idx + 1}`}
                                                style={{
                                                    width: idx === tutorialSlideIndex ? 22 : 8,
                                                    height: 8, borderRadius: 99, border: 'none', cursor: 'pointer',
                                                    background: idx === tutorialSlideIndex ? '#2EA84A' : 'rgba(46,168,74,0.25)',
                                                    transition: 'width 0.25s ease, background 0.2s', padding: 0,
                                                }}
                                            />
                                        ))}
                                    </div>

                                    <button onClick={goToNextTutorialSlide} style={{
                                        background: 'none', border: '1px solid rgba(46,168,74,0.35)',
                                        borderRadius: 8, cursor: 'pointer', padding: '6px 10px',
                                        color: 'var(--text-secondary, #8a9)', display: 'flex', alignItems: 'center',
                                    }} aria-label="Next step">
                                        <ChevronRight size={16} />
                                    </button>
                                </div>

                                {/* Mascot narrator */}
                                <div style={{
                                    display: 'flex', gap: 14, alignItems: 'flex-start',
                                    padding: '16px 20px', margin: '0 16px 16px',
                                    background: 'rgba(46,125,50,0.08)',
                                    border: '1px solid rgba(46,168,74,0.2)',
                                    borderRadius: 12,
                                }}>
                                    <div style={{ flexShrink: 0 }}>
                                        <TradingTurtleMascot speaking={isNarrating} size={80} />
                                    </div>
                                    <div style={{ flex: 1 }}>
                                        <p style={{
                                            margin: '0 0 4px', fontSize: 10, fontWeight: 700,
                                            letterSpacing: '0.1em', color: '#2EA84A', textTransform: 'uppercase',
                                        }}>
                                            Shelly says
                                        </p>
                                        <p style={{
                                            margin: 0, fontSize: 13, lineHeight: 1.65,
                                            color: 'var(--text-primary, #e2e8f0)',
                                            minHeight: 60, fontFamily: 'monospace',
                                        }}>
                                            {typedNarration}
                                            <span style={{
                                                display: 'inline-block', width: 2, height: '1em',
                                                background: isNarrating ? '#2EA84A' : 'transparent',
                                                marginLeft: 2, verticalAlign: 'middle',
                                                animation: isNarrating ? 'blinkCursor 0.7s step-end infinite' : 'none',
                                            }} />
                                        </p>
                                    </div>
                                </div>

                                {/* Footer */}
                                <div style={{
                                    padding: '12px 20px 18px', display: 'flex',
                                    justifyContent: 'space-between', alignItems: 'center',
                                    borderTop: '1px solid rgba(46,168,74,0.12)',
                                }}>
                                    <button
                                        onClick={() => setShowIntro(true)}
                                        style={{
                                            background: 'none', border: 'none', cursor: 'pointer',
                                            fontSize: 12, color: 'var(--text-muted, #64748b)',
                                            padding: '4px 0',
                                        }}
                                    >
                                        ← Back to intro
                                    </button>
                                    <button
                                        onClick={closeTutorial}
                                        style={{
                                            background: '#2EA84A', color: '#fff',
                                            border: 'none', borderRadius: 8,
                                            padding: '9px 24px', fontSize: 14, fontWeight: 600,
                                            cursor: 'pointer', letterSpacing: '0.02em',
                                        }}
                                    >
                                        Get Started →
                                    </button>
                                </div>
                            </>
                        )}
                    </div>

                    <style>{`@keyframes blinkCursor { 0%,100%{opacity:1} 50%{opacity:0} }`}</style>
                </div>
            )}
        </div>
    );
}