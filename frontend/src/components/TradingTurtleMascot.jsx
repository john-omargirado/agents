import React from 'react';

/**
 * TradingTurtleMascot
 * Props:
 *   speaking  {boolean}  — true while isNarrating is true
 *   size      {number}   — rendered width/height in px (default 120)
 */
export default function TradingTurtleMascot({ speaking = false, thinking = false, size = 120 }) {
    return (speaking || thinking) ? <SpeakingTurtle size={size} /> : <IdleTurtle size={size} />;
}

/* ─── SHARED KEYFRAMES ────────────────────────────────────────────────── */
const SHARED_CSS = `
    @keyframes float      { 0%,100%{transform:translateY(0)}  50%{transform:translateY(-5px)} }
    @keyframes legL       { 0%,100%{transform:rotate(0deg)}   50%{transform:rotate(12deg)}  }
    @keyframes legR       { 0%,100%{transform:rotate(0deg)}   50%{transform:rotate(-12deg)} }
    @keyframes armWave    { 0%,100%{transform:rotate(0deg)}   40%{transform:rotate(-18deg)} 70%{transform:rotate(6deg)} }
    @keyframes blink      { 0%,88%,100%{transform:scaleY(1)}  94%{transform:scaleY(0.08)} }
    @keyframes shellGlow  { 0%,100%{opacity:0.12} 50%{opacity:0.32} }
    @keyframes coinSpin   { 0%{transform:rotateY(0deg)} 100%{transform:rotateY(360deg)} }
    @keyframes arrowBounce{ 0%,100%{transform:translateY(0)}  50%{transform:translateY(-5px)} }
    @keyframes mouthAnim  { 0%,100%{transform:scaleY(1)}      50%{transform:scaleY(1.8)} }
    @keyframes armUp      { 0%,100%{transform:rotate(0deg)}   50%{transform:rotate(-30deg)} }
    @keyframes dot1 { 0%,100%{opacity:0.15} 30%{opacity:1} }
    @keyframes dot2 { 0%,100%{opacity:0.15} 50%{opacity:1} }
    @keyframes dot3 { 0%,100%{opacity:0.15} 70%{opacity:1} }
`;

/* ─── IDLE ──────────────────────────────────────────────────────────────── */
function IdleTurtle({ size }) {
    return (
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 160 160"
            shapeRendering="crispEdges" width={size} height={size}
            aria-label="Shelly the forex turtle mascot — idle" role="img"
            style={{ imageRendering: 'pixelated', display: 'block' }}>
            <style>{SHARED_CSS}</style>

            {/* Arrow — static */}
            <rect x="72" y="4" width="16" height="8" fill="#2EA84A" />
            <rect x="64" y="12" width="32" height="8" fill="#2EA84A" />
            <rect x="56" y="20" width="48" height="8" fill="#2EA84A" />
            <rect x="72" y="28" width="16" height="8" fill="#2EA84A" />

            {/* Body group — floats */}
            <g style={{ animation: 'float 2.8s ease-in-out infinite', transformOrigin: '80px 100px' }}>

                {/* Shell */}
                <rect x="24" y="36" width="112" height="84" fill="#2E7D32" />
                <rect x="20" y="112" width="120" height="8" fill="#1B5E20" />

                {/* Shell hex-pattern shimmer (3 subtle lighter patches) */}
                <rect x="44" y="48" width="20" height="18" rx="3" fill="#3A9941" style={{ animation: 'shellGlow 3.2s 0.0s ease-in-out infinite' }} />
                <rect x="70" y="52" width="20" height="18" rx="3" fill="#3A9941" style={{ animation: 'shellGlow 3.2s 0.6s ease-in-out infinite' }} />
                <rect x="96" y="48" width="20" height="18" rx="3" fill="#3A9941" style={{ animation: 'shellGlow 3.2s 1.2s ease-in-out infinite' }} />
                <rect x="56" y="72" width="20" height="18" rx="3" fill="#3A9941" style={{ animation: 'shellGlow 3.2s 0.3s ease-in-out infinite' }} />
                <rect x="82" y="72" width="20" height="18" rx="3" fill="#3A9941" style={{ animation: 'shellGlow 3.2s 0.9s ease-in-out infinite' }} />

                {/* Head outline */}
                <rect x="40" y="44" width="80" height="8" fill="#1A1A1A" />
                {/* Head skin */}
                <rect x="40" y="52" width="80" height="24" fill="#81C784" />

                {/* Eyes with blink + pupil glint */}
                <g style={{ animation: 'blink 4.5s ease-in-out infinite', transformOrigin: '56px 68px' }}>
                    <rect x="52" y="60" width="8" height="16" fill="#1A1A1A" />
                    {/* glint */}
                    <rect x="53" y="61" width="3" height="3" fill="#ffffff" style={{ opacity: 0.7 }} />
                </g>
                <g style={{ animation: 'blink 4.5s 0.2s ease-in-out infinite', transformOrigin: '104px 68px' }}>
                    <rect x="100" y="60" width="8" height="16" fill="#1A1A1A" />
                    <rect x="101" y="61" width="3" height="3" fill="#ffffff" style={{ opacity: 0.7 }} />
                </g>

                {/* Mouth line */}
                <rect x="32" y="76" width="96" height="8" fill="#1A1A1A" />

                {/* Lower body */}
                <rect x="32" y="84" width="96" height="32" fill="#81C784" />

                {/* Left arm — waves */}
                <g style={{ animation: 'armWave 3.6s ease-in-out infinite', transformOrigin: '24px 100px' }}>
                    <rect x="16" y="92" width="16" height="16" fill="#81C784" />
                </g>
                {/* Right arm — waves offset */}
                <g style={{ animation: 'armWave 3.6s 1.8s ease-in-out infinite', transformOrigin: '136px 100px' }}>
                    <rect x="128" y="92" width="16" height="16" fill="#81C784" />
                </g>

                {/* Left leg waddle */}
                <g style={{ animation: 'legL 1.2s ease-in-out infinite', transformOrigin: '56px 116px' }}>
                    <rect x="48" y="116" width="16" height="16" fill="#81C784" />
                </g>
                {/* Right leg waddle */}
                <g style={{ animation: 'legR 1.2s ease-in-out infinite', transformOrigin: '104px 116px' }}>
                    <rect x="96" y="116" width="16" height="16" fill="#81C784" />
                </g>

                {/* Coin */}
                <g style={{ animation: 'coinSpin 2s linear infinite', transformOrigin: '148px 116px', perspective: '80px' }}>
                    <rect x="140" y="108" width="16" height="16" fill="#FBC02D" />
                    <rect x="144" y="104" width="8" height="4" fill="#FBC02D" />
                    <rect x="144" y="124" width="8" height="4" fill="#FBC02D" />
                    <rect x="136" y="108" width="4" height="16" fill="#FBC02D" />
                    <rect x="156" y="108" width="4" height="16" fill="#FBC02D" />
                    <rect x="146" y="108" width="4" height="2" fill="#F57F17" />
                    <rect x="144" y="110" width="2" height="2" fill="#F57F17" />
                    <rect x="146" y="112" width="4" height="2" fill="#F57F17" />
                    <rect x="148" y="114" width="2" height="2" fill="#F57F17" />
                    <rect x="146" y="116" width="4" height="2" fill="#F57F17" />
                </g>
            </g>
        </svg>
    );
}

/* ─── SPEAKING ────────────────────────────────────────────────────────────── */
function SpeakingTurtle({ size }) {
    return (
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 160 160"
            shapeRendering="crispEdges" width={size} height={size}
            aria-label="Shelly the forex turtle mascot — speaking" role="img"
            style={{ imageRendering: 'pixelated', display: 'block' }}>
            <style>{SHARED_CSS}</style>

            {/* Bouncing arrow */}
            <g style={{ animation: 'arrowBounce 0.5s ease-in-out infinite', transformOrigin: '80px 20px' }}>
                <rect x="72" y="4" width="16" height="8" fill="#2EA84A" />
                <rect x="64" y="12" width="32" height="8" fill="#2EA84A" />
                <rect x="56" y="20" width="48" height="8" fill="#2EA84A" />
                <rect x="72" y="28" width="16" height="8" fill="#2EA84A" />
            </g>

            {/* Speech bubble dots (left side) */}
            <circle cx="22" cy="78" r="5" fill="white" style={{ animation: 'dot1 0.8s ease-in-out infinite' }} />
            <circle cx="12" cy="88" r="4" fill="white" style={{ animation: 'dot2 0.8s ease-in-out infinite' }} />
            <circle cx="5" cy="96" r="3" fill="white" style={{ animation: 'dot3 0.8s ease-in-out infinite' }} />

            {/* Body — gentle speaking bounce */}
            <g style={{ animation: 'float 0.6s ease-in-out infinite', transformOrigin: '80px 100px' }}>

                {/* Shell */}
                <rect x="24" y="36" width="112" height="84" fill="#2E7D32" />
                <rect x="20" y="112" width="120" height="8" fill="#1B5E20" />

                {/* Shell shimmer — faster pulse when speaking */}
                <rect x="44" y="48" width="20" height="18" rx="3" fill="#3A9941" style={{ animation: 'shellGlow 1.1s 0.0s ease-in-out infinite' }} />
                <rect x="70" y="52" width="20" height="18" rx="3" fill="#3A9941" style={{ animation: 'shellGlow 1.1s 0.2s ease-in-out infinite' }} />
                <rect x="96" y="48" width="20" height="18" rx="3" fill="#3A9941" style={{ animation: 'shellGlow 1.1s 0.4s ease-in-out infinite' }} />
                <rect x="56" y="72" width="20" height="18" rx="3" fill="#3A9941" style={{ animation: 'shellGlow 1.1s 0.1s ease-in-out infinite' }} />
                <rect x="82" y="72" width="20" height="18" rx="3" fill="#3A9941" style={{ animation: 'shellGlow 1.1s 0.3s ease-in-out infinite' }} />

                {/* Head outline */}
                <rect x="40" y="44" width="80" height="8" fill="#1A1A1A" />
                {/* Head skin */}
                <rect x="40" y="52" width="80" height="24" fill="#81C784" />

                {/* Wide excited eyes */}
                <rect x="50" y="56" width="14" height="20" fill="#1A1A1A" />
                <rect x="51" y="57" width="4" height="4" fill="#ffffff" style={{ opacity: 0.75 }} />
                <rect x="96" y="56" width="14" height="20" fill="#1A1A1A" />
                <rect x="97" y="57" width="4" height="4" fill="#ffffff" style={{ opacity: 0.75 }} />

                {/* Mouth line */}
                <rect x="32" y="76" width="96" height="8" fill="#1A1A1A" />

                {/* Animated open mouth */}
                <rect x="62" y="84" width="36" height="8" fill="#1A1A1A"
                    style={{ animation: 'mouthAnim 0.35s ease-in-out infinite', transformOrigin: '80px 88px' }} />
                <rect x="66" y="86" width="28" height="5" fill="#E57373"
                    style={{ animation: 'mouthAnim 0.35s ease-in-out infinite', transformOrigin: '80px 88px' }} />

                {/* Lower body */}
                <rect x="32" y="84" width="96" height="32" fill="#81C784" />

                {/* Arms raised while speaking */}
                <g style={{ animation: 'armUp 0.6s ease-in-out infinite', transformOrigin: '24px 92px' }}>
                    <rect x="16" y="84" width="16" height="16" fill="#81C784" />
                </g>
                <g style={{ animation: 'armUp 0.6s 0.3s ease-in-out infinite', transformOrigin: '136px 92px' }}>
                    <rect x="128" y="84" width="16" height="16" fill="#81C784" />
                </g>

                {/* Legs (static while speaking) */}
                <rect x="48" y="116" width="16" height="16" fill="#81C784" />
                <rect x="96" y="116" width="16" height="16" fill="#81C784" />

                {/* Coin pulsing */}
                <g style={{ animation: 'shellGlow 1.1s ease-in-out infinite', transformOrigin: '148px 116px' }}>
                    <rect x="140" y="108" width="16" height="16" fill="#FBC02D" />
                    <rect x="144" y="104" width="8" height="4" fill="#FBC02D" />
                    <rect x="144" y="124" width="8" height="4" fill="#FBC02D" />
                    <rect x="136" y="108" width="4" height="16" fill="#FBC02D" />
                    <rect x="156" y="108" width="4" height="16" fill="#FBC02D" />
                    <rect x="146" y="108" width="4" height="2" fill="#F57F17" />
                    <rect x="144" y="110" width="2" height="2" fill="#F57F17" />
                    <rect x="146" y="112" width="4" height="2" fill="#F57F17" />
                    <rect x="148" y="114" width="2" height="2" fill="#F57F17" />
                    <rect x="146" y="116" width="4" height="2" fill="#F57F17" />
                </g>
            </g>
        </svg>
    );
}