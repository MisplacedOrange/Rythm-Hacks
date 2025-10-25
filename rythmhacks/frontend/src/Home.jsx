import React from 'react'
import './Home.css'

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
            No coding required—just drag, drop, and analyze patient data in real-time.
          </p>
          <div className="hero-actions">
            <button className="btn-primary">Get Started</button>
            <button className="btn-secondary">Watch Demo</button>
          </div>
        </section>

        {/* Features Section */}
        <section className="features-section">
          <div className="feature-grid">
            <div className="feature-item">
              <h3 className="feature-title">Interactive Dashboards</h3>
              <p className="feature-description">
                Create stunning, interactive dashboards with patient metrics, 
                treatment outcomes, and population health data—no coding required.
              </p>
            </div>

            <div className="feature-item">
              <h3 className="feature-title">Real-Time Analytics</h3>
              <p className="feature-description">
                Monitor vital statistics, track patient trends, and visualize 
                clinical data in real-time with intuitive graphs and charts.
              </p>
            </div>

            <div className="feature-item">
              <h3 className="feature-title">Custom Reports</h3>
              <p className="feature-description">
                Generate comprehensive reports with drag-and-drop simplicity. 
                Export publication-ready visualizations instantly.
              </p>
            </div>

            <div className="feature-item">
              <h3 className="feature-title">Secure & Compliant</h3>
              <p className="feature-description">
                HIPAA-compliant infrastructure with enterprise-grade security. 
                Your patient data stays protected and private.
              </p>
            </div>
          </div>
        </section>

        {/* Stats Section */}
        <section className="stats-section">
          <div className="stats-grid">
            <div className="stat-item">
              <div className="stat-number">5,000+</div>
              <div className="stat-label">Healthcare Professionals</div>
            </div>
            <div className="stat-item">
              <div className="stat-number">100%</div>
              <div className="stat-label">HIPAA Compliant</div>
            </div>
            <div className="stat-item">
              <div className="stat-number">10M+</div>
              <div className="stat-label">Visualizations Created</div>
            </div>
          </div>
        </section>

        {/* CTA Section */}
        <section className="cta-section">
          <h2 className="cta-title">Ready to Transform Your Healthcare Data?</h2>
          <p className="cta-description">
            Join thousands of healthcare professionals visualizing data without writing a single line of code.
          </p>
          <button className="btn-primary">Start Free Trial</button>
        </section>
      </div>
    </div>
  )
}

export default Home
