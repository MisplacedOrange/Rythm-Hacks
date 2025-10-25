import React from 'react'
import { Link } from 'react-router-dom'
import './Footer.css'
import logoIcon from '../assets/images/MediLytica.png'

const Footer = () => {
  return (
    <footer className="site-footer">
      <div className="footer-container">
        <div className="footer-top">
          <div className="footer-brand">
            <Link to="/" className="brand-link">
              <span className="brand-logo">
                <img src={logoIcon} alt="MediLytica" />
              </span>
              <span className="brand-name">MediLytica</span>
            </Link>
            <p className="brand-tagline">No-code data visualization tools for busy healthcare professionals.</p>
          </div>

          <div className="footer-columns">
            <div className="footer-col">
              <h4 className="footer-title">Explore</h4>
              <ul className="footer-links">
                <li><Link to="/">Home</Link></li>
                <li><Link to="/dashboard">Dashboard</Link></li>
                <li><Link to="/contact">Contact</Link></li>
              </ul>
            </div>

            <div className="footer-col">
              <h4 className="footer-title">Resources</h4>
              <ul className="footer-links">
                <li><a href="https://github.com/nix-life/Rythm-Hacks">GitHub</a></li>
              </ul>
            </div>

            <div className="footer-col">
              <h4 className="footer-title">Get in touch</h4>
              <ul className="footer-links">
                <li><a href="mailto:support@medilytica.com">support@medilytica.com</a></li>
                <li><a href="tel:+18006334328">1-437-499-4754</a></li>
              </ul>
            </div>
          </div>
        </div>

        <div className="footer-bottom">
          <p className="copyright">© {new Date().getFullYear()} MediLytica. All rights reserved.</p>
          <div className="footer-legal">
            <a href="#privacy">Privacy</a>
            <a href="#terms">Terms</a>
          </div>
        </div>
      </div>
    </footer>
  )
}

export default Footer