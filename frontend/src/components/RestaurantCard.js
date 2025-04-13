import React from 'react';

function RestaurantCard({ restaurant }) {
  return (
    <div style={styles.card}>
      <h2>{restaurant.name}</h2>
      <p><strong>Type:</strong> {restaurant.category}</p>
      <p><strong>Rating:</strong> ⭐ {restaurant.rating}</p>
      <p><strong>Price:</strong> {'$'.repeat(restaurant.price)}</p>
      <p><strong>Address:</strong> {restaurant.address}</p>
    </div>
  );
}

const styles = {
  card: {
    background: 'white',
    borderRadius: '10px',
    padding: '1rem',
    margin: '1rem',
    width: '280px',
    boxShadow: '0 4px 12px rgba(0,0,0,0.1)',
    textAlign: 'left',
    lineHeight: '1.5rem'
  }
};

export default RestaurantCard;
