import React from 'react'
import Navbar from './components/Navbar'
import Home from './pages/Home'

function App() {
  return (
    <>
      <Navbar />
      <div className="container mt-4">
        <Home />
      </div>
    </>
  )
}

export default App