import React from 'react'

interface ScrollableProps {
  children: React.ReactNode
  className?: string
  id?: string
}

export const Scrollable: React.FC<ScrollableProps> = ({ 
  children, 
  className = '', 
  id 
}) => {
  return (
    <div 
      id={id}
      className={`overflow-y-auto ${className}`}
      style={{ 
        scrollBehavior: 'smooth',
        WebkitOverflowScrolling: 'touch'
      }}
    >
      {children}
    </div>
  )
}

export default Scrollable