import React, { useState } from 'react'
import { Link } from 'react-router-dom'
import './Navbar.css'
import logoIcon from '../assets/m.svg'

const Navbar = () => {
  const [isMenuOpen, setIsMenuOpen] = useState(false)

  return (
    <nav className="navbar">
      <div className="navbar-container">
        <div className="navbar-content">
          <div className="navbar-brand">
            <Link to="/" className="navbar-logo">
              <div className="logo-icon">
                <img src={logoIcon} alt="MediLytica Logo" />
              </div>
              <span className="logo-text">MediLytica</span>
            </Link>
          </div>
          <div className="navbar-menu">
            <Link to="/">Features</Link>
            <a href="#dashboard">Dashboard</a>
            <Link to="/contact">Contact</Link>
          </div>
          <div className="navbar-auth">
            <button className="auth-login">Log in</button>
            <button className="auth-signup">Sign up</button>
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
            <Link to="/" onClick={() => setIsMenuOpen(false)}>Product</Link>
            <a href="#dashboard" onClick={() => setIsMenuOpen(false)}>Dashboard</a>
            <Link to="/contact" onClick={() => setIsMenuOpen(false)}>Contact</Link>
          </div>
        </div>
      )}
    </nav>
  )
}

export default Navbar
