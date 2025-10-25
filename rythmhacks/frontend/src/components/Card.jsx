import React from 'react';
import './Card.css'

const Card = ({ title, description, icon, image }) => {
  return (
    <div className="card">
      <div className="card-content">
        {!image && icon && <div className="card-icon">{icon}</div>}
        <h3 className="card-title">{title}</h3>
        <p className="card-description">{description}</p>
        {image && (
          <div className="card-image">
            <img src={image} alt={title} />
          </div>
        )}
      </div>
      <div className="card-expand">
        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <line x1="12" y1="5" x2="12" y2="19"></line>
          <line x1="5" y1="12" x2="19" y2="12"></line>
        </svg>
      </div>
    </div>
  );
}

export default Card;
