import React from 'react';

/**
 * Reusable metrics section component for container + label pattern
 * Provides consistent spacing and styling for metric groupings
 */
export default function MetricsSection({ title, children, className = '' }) {
    return (
        <div className={`metrics-section ${className}`}>
            {title && <div className="metrics-label">{title}</div>}
            {children}
        </div>
    );
}
