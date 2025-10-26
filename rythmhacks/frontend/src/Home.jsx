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
            Turn your data into actionable insights and easily visualized graphical data
          </h1>
          <p className="hero-description">
            Empower your machine learning workflow with powerful data visualization tools. 
            Gain insight collaboratively and analyze your data in a way that can be leveraged 
            by ML engineers and researchers from other fields.
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
              description="Create stunning informative dashboards. Visualize complex datasets and make decisions."
            />
            
            <Card 
              title="Collaborate with Others"
              description="Using our chat feature and code editor, work together in real time and develop models at a rapid pace."
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
                  Bring professionals from multiple different fields, analysts, and administrators together around the same source of truth.
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

        <section className="stats-section">
          <div className="stats-grid">
            <div className="stat-item">
              <div className="stat-number">0</div>
              <div className="stat-label">Professionals from different fields</div>
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
