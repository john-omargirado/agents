// ContextualHelp.jsx
import React, { useState, useRef, useEffect } from 'react';
import { createPortal } from 'react-dom';
import { HelpCircle, X } from 'lucide-react';

const HELP_CONTENT = {
    pair: {
        beginner: {
            headline: 'What is a forex pair?',
            body: "Forex (foreign exchange) means swapping one currency for another — like converting dollars to euros at the airport. A 'pair' is just the two currencies you're comparing. EUR/USD means: how many US dollars does one euro buy?\n\nIf EUR/USD = 1.08, one euro buys $1.08.",
            analogy: '🧳 Think of it like airport currency exchange — you hand in one currency and receive another.',
        },
        basic: {
            headline: 'Forex pair selection',
            body: "Major pairs (EUR/USD, USD/JPY etc.) have the tightest spreads and most liquidity — best for analysis accuracy. Exotic pairs like USD/PHP have wider spreads and less predictable news data.",
        },
        intermediate: {
            headline: 'Pair selection',
            body: "Majors dominate liquidity and CE news coverage. PHP exotics use real data but expect wider spreads and lower CE article counts.",
        },
    },

    capital: {
        beginner: {
            headline: 'What is account capital?',
            body: "This is how much money you're imagining you have in your trading account. The system uses it to calculate position size.\n\nStart with $1,000 as a safe practice number.",
            analogy: '💰 Like setting a budget before shopping.',
        },
        basic: {
            headline: 'Account capital',
            body: "Used for lot sizing. Risk % × capital determines max loss per trade.",
        },
        intermediate: {
            headline: 'Capital',
            body: "Risk $ = capital × riskThreshold. Lot size derived from SL distance and pip value.",
        },
    },

    leverage: {
        beginner: {
            headline: 'What is leverage?',
            body: "Leverage lets you control more money than you actually have. 1:10 means $1 controls $10.\n\nFor learning, use 1:1.",
            analogy: '🏋️ Like a lever — small force, big movement.',
        },
        basic: {
            headline: 'Leverage',
            body: "Higher leverage increases both profit and loss exposure.",
        },
        intermediate: {
            headline: 'Leverage',
            body: "Affects margin only, not risk calculation logic.",
        },
    },

    risk: {
        beginner: {
            headline: 'What is risk per trade?',
            body: "This is how much you're willing to lose per trade.\n\n1% risk on $1,000 = $10 loss max.",
            analogy: '🎯 Like setting a limit before playing.',
        },
        basic: {
            headline: 'Risk threshold',
            body: "Controls position sizing based on account size.",
        },
        intermediate: {
            headline: 'Risk threshold',
            body: "Direct input into lot size calculation using ATR stop distance.",
        },
    },

    tts: {
        beginner: {
            headline: 'What is TTS?',
            body: "It reads charts and decides if price looks bullish or bearish.",
            analogy: '📊 Like reading a heartbeat monitor.',
        },
        basic: {
            headline: 'TTS',
            body: "Combines indicators into a single trend score.",
        },
        intermediate: {
            headline: 'TTS score',
            body: "Weighted technical signal (trend + momentum + volatility).",
        },
    },

    ce: {
        beginner: {
            headline: 'What is CE?',
            body: "It reads financial news and measures sentiment.",
            analogy: '📰 Like scanning headlines for mood.',
        },
        basic: {
            headline: 'CE agent',
            body: "News sentiment scoring system for currency pairs.",
        },
        intermediate: {
            headline: 'CE score',
            body: "Normalized sentiment score weighted by confidence.",
        },
    },

    siv: {
        beginner: {
            headline: 'What is SIV?',
            body: "It checks if chart data and news data match before allowing trades.",
            analogy: '🔍 Like a referee checking fairness.',
        },
        basic: {
            headline: 'SIV',
            body: "Prevents trades when data sources disagree.",
        },
        intermediate: {
            headline: 'SIV multiplier',
            body: "Blocks or adjusts signals based on data integrity.",
        },
    },

    verdict: {
        beginner: {
            headline: 'What is Verdict?',
            body: "Final decision: BUY, SELL, or HOLD based on all agents.",
            analogy: '⚖️ Like a judge deciding a case.',
        },
        basic: {
            headline: 'Verdict',
            body: "Weighted combination of all signals.",
        },
        intermediate: {
            headline: 'Verdict system',
            body: "Final ensemble decision with risk filtering.",
        },
    },
};

export default function ContextualHelp({ topic, experienceLevel }) {
    const [pos, setPos] = useState(null);
    const [open, setOpen] = useState(false);
    const btnRef = useRef(null);
    const panelRef = useRef(null); // Ref for the popup panel

    const level = experienceLevel || 'beginner';
    const content = HELP_CONTENT[topic]?.[level] || HELP_CONTENT[topic]?.beginner;

    // Handle clicking outside and scrolling
    // ContextualHelp.jsx - Updated scroll handling logic
    useEffect(() => {
        if (!open) return;

        const handleOutsideClick = (e) => {
            if (
                panelRef.current?.contains(e.target) ||
                btnRef.current?.contains(e.target)
            ) {
                return;
            }
            setOpen(false);
        };

        const handleScroll = (e) => {
            // Only close if the scroll event target is NOT inside our panel
            // This allows users to scroll long help text without the panel vanishing
            if (panelRef.current && panelRef.current.contains(e.target)) {
                return;
            }
            setOpen(false);
        };

        document.addEventListener('mousedown', handleOutsideClick);
        // Use capture phase (true) to detect scrolls on any element in the document
        window.addEventListener('scroll', handleScroll, true);

        return () => {
            document.removeEventListener('mousedown', handleOutsideClick);
            window.removeEventListener('scroll', handleScroll, true);
        };
    }, [open]);

    if (!content) return null;

    const handleToggle = () => {
        if (!open && btnRef.current) {
            const rect = btnRef.current.getBoundingClientRect();
            let left = rect.left;

            const padding = 10;
            const viewportWidth = window.innerWidth;
            const estimatedWidth = 260;

            if (left + estimatedWidth > viewportWidth - padding) {
                left = viewportWidth - estimatedWidth - padding;
            }

            if (left < padding) left = padding;

            setPos({
                top: rect.bottom + 8,
                left,
            });
        }

        setOpen(v => !v);
    };

    return (
        <span className="ctx-help-wrap">
            <button
                ref={btnRef}
                className="ctx-help-btn"
                type="button"
                aria-expanded={open}
                onClick={handleToggle}
            >
                <HelpCircle size={13} />
            </button>

            {open && pos && createPortal(
                <div
                    ref={panelRef}
                    className="ctx-help-panel"
                    style={{
                        position: 'fixed',
                        top: pos.top,
                        left: pos.left,
                        zIndex: 999999,
                    }}
                    role="tooltip"
                >
                    <div className="ctx-help-header">
                        <span className="ctx-help-headline">
                            {content.headline}
                        </span>

                        <button
                            className="ctx-help-close"
                            onClick={() => setOpen(false)}
                        >
                            <X size={12} />
                        </button>
                    </div>

                    <p className="ctx-help-body">{content.body}</p>

                    {content.analogy && (
                        <p className="ctx-help-analogy">
                            {content.analogy}
                        </p>
                    )}
                </div>,
                document.body
            )}
        </span>
    );
}