import React from 'react';

/**
 * Reusable metrics row component for consistent formatting
 * Eliminates repetitive metrics-row/metrics-label/metrics-value patterns
 */
export default function MetricsRow({ label, value, valueClass = '' }) {
    return (
        <div className="metrics-row">
            <div className="metrics-label">{label}</div>
            <div className={`metrics-value ${valueClass}`}>{value}</div>
        </div>
    );
}
