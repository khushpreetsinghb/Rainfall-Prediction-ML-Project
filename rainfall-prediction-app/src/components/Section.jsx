import React from 'react'

const Section = ({ title, children }) => {
    return (
        <div className="mb-5">
            <h3 className="text-primary border-bottom pb-2 mb-3">{title}</h3>
            {children}
        </div>
    )
}

export default Section