import React, { useState } from 'react';
import InputForm from './components/InputForm';
import RestaurantCard from './components/RestaurantCard';
import './App.css';

function App() {
  const [restaurants, setRestaurants] = useState([]);
  const [loading, setLoading] = useState(false);

  const handleSearch = async (filters) => {
    setLoading(true);
    // Placeholder for now — simulate delay
    setTimeout(() => {
      setRestaurants([
        {
          name: 'Sunset Sushi',
          category: 'Japanese',
          rating: 4.6,
          price: 2,
          address: '123 Tokyo Lane',
        },
        {
          name: 'Comfort Bites',
          category: 'American',
          rating: 4.2,
          price: 1,
          address: '456 Cozy Ave',
        },
      ]);
      setLoading(false);
    }, 1500);
  };

  return (
    <div className="App">
      <h1>Restaurant Recommender</h1>
      <InputForm onSearch={handleSearch} />

      {loading && <p style={styles.loading}>🍽️ Finding your perfect spot...</p>}

      <div style={styles.results}>
        {restaurants.map((r, i) => (
          <RestaurantCard key={i} restaurant={r} />
        ))}
      </div>
    </div>
  );
}

const styles = {
  results: {
    display: 'flex',
    justifyContent: 'center',
    flexWrap: 'wrap',
    padding: '1rem',
  },
  loading: {
    fontSize: '1.2rem',
    textAlign: 'center',
    marginTop: '2rem',
  }
};

export default App;
