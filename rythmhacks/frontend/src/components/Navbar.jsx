import React, { useState } from 'react'
import { Link } from 'react-router-dom'
import './Navbar.css'
import logoIcon from '../assets/images/MediLytica.png'

const Navbar = () => {
  const [isMenuOpen, setIsMenuOpen] = useState(false)

  return (
    <nav className="navbar">
      <div className="navbar-container">
        <div className="navbar-content">
          <div className="navbar-brand">
            <Link to="/" className="navbar-logo">
              <div className="logo-icon">
                <img src={logoIcon} alt="Medilytica Logo" />
              </div>
              <span className="logo-text">Medilytica</span>
            </Link>
          </div>
          <div className="navbar-menu">
            <Link to="/">Home</Link>
            <Link to="/dashboard">Features</Link>
            <Link to="/contact">Contact Us</Link>
          </div>
          <div className="navbar-auth">
            <button className="auth-button">button</button>
          </div>
          <button onClick={() => setIsMenuOpen(!isMenuOpen)} className="mobile-menu-button">
            <svg stroke="currentColor" fill="none" viewBox="0 0 24 24">
              {isMenuOpen ? (
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M6 18L18 6M6 6l12 12" />
              ) : (
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 6h16M4 12h16M4 18h16" />
              )}
            </svg>
          </button>
        </div>
      </div>
      {isMenuOpen && (
        <div className="mobile-menu">
          <div className="mobile-menu-content">
            <Link to="/" onClick={() => setIsMenuOpen(false)}>Home</Link>
            <Link to="/dashboard" onClick={() => setIsMenuOpen(false)}>Features</Link>
            <Link to="/contact" onClick={() => setIsMenuOpen(false)}>Contact Us</Link>
          </div>
        </div>
      )}
    </nav>
  )
}

export default Navbar
