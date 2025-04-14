import React, { useState } from 'react';
import InputForm from './components/InputForm';
import RestaurantCard from './components/RestaurantCard';
import './App.css';

function App() {
  const [restaurants, setRestaurants] = useState([]);
  const [loading, setLoading] = useState(false);

  const handleSearch = async (filters) => {
    setLoading(true);
    try {
      const res = await fetch("https://module3projectbackend.onrender.com/recommend", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(filters),
      });
      const data = await res.json();
      setRestaurants(data || []);
    } catch (error) {
      console.error("Error fetching recommendations:", error);
    } finally {
      setLoading(false);
    }
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
