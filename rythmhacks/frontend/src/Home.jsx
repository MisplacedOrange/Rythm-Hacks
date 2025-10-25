import React from 'react'
import './Home.css'

const Home = () => {
  return (
    <div className="landing-page">
      <div className="landing-container">

        <section className="hero-section">
          <h1 className="hero-title">
            Plan and build your product
          </h1>
          <p className="hero-description">
            A purpose-built tool for modern product development. 
            Streamline your workflow from roadmap to release.
          </p>
          <div className="hero-actions">
            <button className="btn-primary">Get Started</button>
            <button className="btn-secondary">Learn More</button>
          </div>
        </section>

        {/* Features Section */}
        <section className="features-section">
          <div className="feature-grid">
            <div className="feature-item">
              <h3 className="feature-title">Issue Tracking</h3>
              <p className="feature-description">
                Create, triage, and manage issues with precision. 
                Keep your team aligned and focused.
              </p>
            </div>

            <div className="feature-item">
              <h3 className="feature-title">Projects</h3>
              <p className="feature-description">
                Organize work into projects and initiatives. 
                Connect daily work to strategic goals.
              </p>
            </div>

            <div className="feature-item">
              <h3 className="feature-title">Cycles</h3>
              <p className="feature-description">
                Time-boxed sprints to maintain momentum. 
                Ship consistently and predictably.
              </p>
            </div>

            <div className="feature-item">
              <h3 className="feature-title">Roadmaps</h3>
              <p className="feature-description">
                Plan from strategic initiatives to execution. 
                Track progress with analytics and insights.
              </p>
            </div>
          </div>
        </section>

        {/* Stats Section */}
        <section className="stats-section">
          <div className="stats-grid">
            <div className="stat-item">
              <div className="stat-number">15,000+</div>
              <div className="stat-label">Organizations</div>
            </div>
            <div className="stat-item">
              <div className="stat-number">99.9%</div>
              <div className="stat-label">Uptime</div>
            </div>
            <div className="stat-item">
              <div className="stat-number">150ms</div>
              <div className="stat-label">Average Response</div>
            </div>
          </div>
        </section>

        {/* CTA Section */}
        <section className="cta-section">
          <h2 className="cta-title">Ready to get started?</h2>
          <p className="cta-description">
            Join thousands of teams building better products.
          </p>
          <button className="btn-primary">Start Building</button>
        </section>
      </div>
    </div>
  )
}

export default Home
