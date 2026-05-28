import React from 'react';
import { TrendingUp, TrendingDown, Minus } from 'lucide-react';

/**
 * Displays a trading signal badge (BUY/SELL/HOLD) with semantic coloring
 * Reusable across backtesting and trading assistant views
 */
export default function DaySignalBadge({ signal }) {
    const s = (signal || 'HOLD').toUpperCase();
    const className = s === 'BUY' ? 'buy' : s === 'SELL' ? 'sell' : 'hold';
    const Icon = s === 'BUY' ? TrendingUp : s === 'SELL' ? TrendingDown : Minus;

    return (
        <span className={`signal-badge ${className}`}>
            <Icon size={18} />
            {s}
        </span>
    );
}
