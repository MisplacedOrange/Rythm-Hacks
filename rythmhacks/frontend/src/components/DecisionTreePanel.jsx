import React from 'react'

export default function DecisionTreePanel() {
  // Simple static SVG tree mock to mirror screenshot layout
  return (
    <div style={{ background:'#fff', border:'1px solid var(--border-subtle)', borderRadius:'8px', padding:'12px', maxHeight: '400px', overflow: 'auto' }}>
      <h3 style={{ marginBottom:'8px' }}>Decision Tree</h3>
      <div className="chart-container">
        <svg viewBox="0 0 600 320" style={{ width:'100%', height:'100%' }}>
          {/* Edges */}
        <line x1="300" y1="60" x2="180" y2="130" stroke="#999" />
        <line x1="300" y1="60" x2="420" y2="130" stroke="#999" />
        <line x1="180" y1="160" x2="120" y2="230" stroke="#999" />
        <line x1="180" y1="160" x2="240" y2="230" stroke="#999" />
        <line x1="420" y1="160" x2="360" y2="230" stroke="#999" />
        <line x1="420" y1="160" x2="480" y2="230" stroke="#999" />

        {/* Nodes */}
        <rect x="250" y="30" rx="6" ry="6" width="100" height="40" fill="#4682B4" opacity="0.85" />
        <text x="300" y="55" textAnchor="middle" fontSize="12" fill="#fff">root: x &lt; 3.1</text>

        <rect x="130" y="130" rx="6" ry="6" width="100" height="40" fill="#F4A460" opacity="0.9" />
        <text x="180" y="155" textAnchor="middle" fontSize="12" fill="#000">x &lt; 1.7</text>

        <rect x="370" y="130" rx="6" ry="6" width="100" height="40" fill="#4682B4" opacity="0.85" />
        <text x="420" y="155" textAnchor="middle" fontSize="12" fill="#fff">x ≥ 1.7</text>

        <rect x="70" y="230" rx="6" ry="6" width="100" height="40" fill="#F4A460" opacity="0.9" />
        <text x="120" y="255" textAnchor="middle" fontSize="12" fill="#000">Class A</text>

        <rect x="200" y="230" rx="6" ry="6" width="100" height="40" fill="#4682B4" opacity="0.85" />
        <text x="250" y="255" textAnchor="middle" fontSize="12" fill="#fff">Class B</text>

        <rect x="330" y="230" rx="6" ry="6" width="100" height="40" fill="#F4A460" opacity="0.9" />
        <text x="380" y="255" textAnchor="middle" fontSize="12" fill="#000">Class A</text>

        <rect x="460" y="230" rx="6" ry="6" width="100" height="40" fill="#4682B4" opacity="0.85" />
        <text x="510" y="255" textAnchor="middle" fontSize="12" fill="#fff">Class B</text>
      </svg>
      </div>
    </div>
  )
}

