import React from 'react'

const OutputBlock = ({ title, output }) => {
    return (
        <div className="card mb-4 border-secondary">
            <div className="card-header bg-light text-dark">{title}</div>
            <div className="card-body output-text">
                <pre style={{ backgroundColor: '#fff', padding: '1rem' }}>
                    <code>{output}</code>
                </pre>
            </div>
        </div>
    )
}

export default OutputBlock
