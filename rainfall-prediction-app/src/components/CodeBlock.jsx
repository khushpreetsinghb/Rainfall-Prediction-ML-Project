import React from 'react'

const CodeBlock = ({ title, code }) => {
    return (
        <div className="card mb-4">
            <div className="card-header bg-dark text-white">{title}</div>
            <div className="card-body">
                <pre style={{ backgroundColor: '#f8f9fa', padding: '1rem' }}>
                    <code>{code}</code>
                </pre>
            </div>
        </div>
    )
}

export default CodeBlock