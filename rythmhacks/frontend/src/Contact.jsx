import React, { useState } from 'react'
import './Contact.css'

const Contact = () => {
  const [formData, setFormData] = useState({
    name: '',
    email: '',
    organization: '',
    message: ''
  })
  const [isSubmitting, setIsSubmitting] = useState(false)

  const handleChange = (e) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value
    })
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    setIsSubmitting(true)

    const webhookUrl = 'https://discord.com/api/webhooks/1431703360142311524/OV7FAGvwaRm463VWKTDDDx92H7L2RIrIfObiJDvVv1jDVw3H0ImXN_ERBdZyD4Czcgpp'

    const discordMessage = {
      embeds: [{
        title: 'New Contact Form Submission',
        color: 0x00B5E2, // Aqua color in hex
        fields: [
          {
            name: '👤 Name',
            value: formData.name,
            inline: true
          },
          {
            name: '📧 Email',
            value: formData.email,
            inline: true
          },
          {
            name: '🏥 Organization',
            value: formData.organization || 'Not provided',
            inline: false
          },
          {
            name: '💬 Message',
            value: formData.message,
            inline: false
          }
        ],
        timestamp: new Date().toISOString(),
        footer: {
          text: 'MediLytica Contact Form'
        }
      }]
    }

    try {
      const response = await fetch(webhookUrl, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(discordMessage)
      })

      if (response.ok) {
        alert('Thank you for contacting us! We will get back to you soon.')
        setFormData({ name: '', email: '', organization: '', message: '' })
      } else {
        alert('There was an error sending your message. Please try again.')
      }
    } catch (error) {
      console.error('Error sending to Discord:', error)
      alert('There was an error sending your message. Please try again.')
    } finally {
      setIsSubmitting(false)
    }
  }

  return (
    <div className="contact-page">
      <div className="contact-container">
        <section className="contact-hero-section">
          <h1 className="contact-hero-title">
            Get In Touch
          </h1>
          <p className="contact-hero-description">
            Have questions about MediLytica? We're here to help healthcare professionals 
            transform their data into actionable insights.
          </p>
        </section>

        <section className="contact-form-section">
          <div className="contact-content-grid">
            <div className="contact-info">
              <h2 className="contact-info-title">Connect Us</h2>
              <p className="contact-info-description">
                Our team is dedicated to supporting healthcare professionals 
                with powerful, easy-to-use data visualization tools.
              </p>
            </div>

            <div className="contact-form-wrapper">
              <form className="contact-form" onSubmit={handleSubmit}>
                <div className="form-group">
                  <label htmlFor="name" className="form-label">Full Name</label>
                  <input
                    type="text"
                    id="name"
                    name="name"
                    value={formData.name}
                    onChange={handleChange}
                    className="form-input"
                    placeholder="Dr. Jane Smith"
                    required
                  />
                </div>

                <div className="form-group">
                  <label htmlFor="email" className="form-label">Email Address</label>
                  <input
                    type="email"
                    id="email"
                    name="email"
                    value={formData.email}
                    onChange={handleChange}
                    className="form-input"
                    placeholder="jane.smith@hospital.com"
                    required
                  />
                </div>

                <div className="form-group">
                  <label htmlFor="organization" className="form-label">Organization</label>
                  <input
                    type="text"
                    id="organization"
                    name="organization"
                    value={formData.organization}
                    onChange={handleChange}
                    className="form-input"
                    placeholder="City General Hospital"
                  />
                </div>

                <div className="form-group">
                  <label htmlFor="message" className="form-label">Message</label>
                  <textarea
                    id="message"
                    name="message"
                    value={formData.message}
                    onChange={handleChange}
                    className="form-textarea"
                    placeholder="Tell us how we can help you..."
                    rows="5"
                    required
                  ></textarea>
                </div>

                <button type="submit" className="btn-primary btn-submit" disabled={isSubmitting}>
                  {isSubmitting ? 'Sending...' : 'Send Message'}
                </button>
              </form>
            </div>
          </div>
        </section>
      </div>
    </div>
  )
}

export default Contact
