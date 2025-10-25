import React from 'react'
import './Home.css'
import Card from './components/Card'
import collaborationImage from './assets/images/collaboration.jpg'

const Home = () => {
  return (
    <div className="landing-page">
      <div className="landing-container">

        <section className="hero-section">
          <h1 className="hero-title">
            Transform Healthcare Data Into Insights
          </h1>
          <p className="hero-description">
            Empower healthcare professionals with powerful data visualization tools. 
            No coding required, just drag, drop, and analyze patient data in real-time.
          </p>
          <div className="hero-actions">
            <button className="btn-primary">Get Started</button>
            <button className="btn-secondary">Watch Demo</button>
          </div>
        </section>

        <section className="features-section">
          <div className="feature-grid">
            <Card 
              title="Interactive Dashboards"
              description="Create stunning, interactive dashboards with patient metrics, treatment outcomes, and population health data—no coding required."
            />
            
            <Card 
              title="Real-Time Analytics"
              description="Monitor vital statistics, track patient trends, and visualize clinical data in real-time with intuitive graphs and charts."
            />
            
            <Card 
              title="Custom Reports"
              description="Generate comprehensive reports with drag-and-drop simplicity. Export publication-ready visualizations instantly."
            />

          </div>
        </section>

        <h2 className="collaboration-title">COLLABORATE WITH PROFESSIONALS</h2>
        <section className="collaboration-section">
          <div className="collaboration-wrapper">
            <div className="collaboration-grid">
              <div className="collaboration-text">
                <h3 className="collaboration-subtitle">Work seamlessly with your team</h3>
                <p className="collaboration-description">
                  Bring clinicians, analysts, and administrators together around the same source of truth.
                  Review patient outcomes, spot trends, and make data-driven decisions in real time using shared dashboards.
                </p>
                <ul className="collaboration-points">
                  <li>Secure sharing of dashboards and insights</li>
                  <li>Commentary and context alongside visuals</li>
                  <li>Unified view of patient and operational metrics</li>
                </ul>
              </div>
              <div className="collaboration-media">
                <img
                  className="collaboration-image"
                  src={collaborationImage}
                  alt="Healthcare team collaborating over a shared analytics dashboard"
                  loading="lazy"
                />
              </div>
            </div>
          </div>
        </section>

        {/* Stats Section */}
        <section className="stats-section">
          <div className="stats-grid">
            <div className="stat-item">
              <div className="stat-number">0</div>
              <div className="stat-label">Healthcare Professionals</div>
            </div>
            <div className="stat-item">
              <div className="stat-number">99%</div>
              <div className="stat-label">Sucessful Build Rate</div>
            </div>
            <div className="stat-item">
              <div className="stat-number">2</div>
              <div className="stat-label">Visualizations Created</div>
            </div>
          </div>
        </section>
      </div>
    </div>
  )
}

export default Home
