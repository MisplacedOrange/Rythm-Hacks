import React, { useState } from 'react'
import './Navbar.css'

const Navbar = () => {
  const [isMenuOpen, setIsMenuOpen] = useState(false)

  const toggleMenu = () => {
    setIsMenuOpen(!isMenuOpen)
  }

  return (
    <nav className="navbar">
      <div className="navbar-container">
        <div className="navbar-content">
          {/* Logo/Brand */}
          <div className="navbar-brand">
            <div className="navbar-logo">
              <div className="logo-icon">
                <svg fill="currentColor" viewBox="0 0 20 20">
                  <path d="M3 4a1 1 0 011-1h12a1 1 0 011 1v2a1 1 0 01-1 1H4a1 1 0 01-1-1V4zM3 10a1 1 0 011-1h6a1 1 0 011 1v6a1 1 0 01-1 1H4a1 1 0 01-1-1v-6zM14 9a1 1 0 00-1 1v6a1 1 0 001 1h2a1 1 0 001-1v-6a1 1 0 00-1-1h-2z" />
                </svg>
              </div>
              <span className="logo-text">MediLytica</span>
            </div>
          </div>

          {/* Desktop Navigation Menu */}
          <div className="navbar-menu">
            <a href="#product">Product</a>
            <a href="#resources">Resources</a>
            <a href="#pricing">Pricing</a>
            <a href="#customers">Customers</a>
            <a href="#now">Now</a>
            <a href="#contact">Contact</a>
          </div>

          {/* Desktop Auth Buttons */}
          <div className="navbar-auth">
            <button className="auth-login">Log in</button>
            <button className="auth-signup">Sign up</button>
          </div>

          {/* Mobile menu button */}
          <button
            onClick={toggleMenu}
            className="mobile-menu-button"
          >
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

      {/* Mobile Navigation Menu */}
      {isMenuOpen && (
        <div className="mobile-menu">
          <div className="mobile-menu-content">
            <a href="#product">Product</a>
            <a href="#resources">Resources</a>
            <a href="#pricing">Pricing</a>
            <a href="#customers">Customers</a>
            <a href="#now">Now</a>
            <a href="#contact">Contact</a>
            <div className="mobile-auth-section">
              <div className="mobile-auth-buttons">
                <button className="mobile-auth-login">Log in</button>
                <button className="mobile-auth-signup">Sign up</button>
              </div>
            </div>
          </div>
        </div>
      )}
    </nav>
  )
}

export default Navbar