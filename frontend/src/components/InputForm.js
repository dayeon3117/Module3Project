import React, { useState } from 'react';

function InputForm({ onSearch }) {
  const [location, setLocation] = useState('');
  const [food, setFood] = useState('');
  const [price, setPrice] = useState('');
  const [model, setModel] = useState('naive');

  const handleSubmit = (e) => {
    e.preventDefault();
    onSearch({ location, food, price, model });
  };

  return (
    <form onSubmit={handleSubmit} style={styles.form}>
      <div style={styles.row}>
        <label style={styles.label}>Location:</label>
        <input
          value={location}
          onChange={(e) => setLocation(e.target.value)}
          placeholder="e.g. North Carolina"
          style={styles.input}
        />

        <label style={styles.label}>Food Type:</label>
        <input
          value={food}
          onChange={(e) => setFood(e.target.value)}
          placeholder="e.g. sushi"
          style={styles.input}
        />

        <label style={styles.label}>Price:</label>
        <select value={price} onChange={(e) => setPrice(e.target.value)} style={styles.select}>
          <option value="">Select</option>
          <option value="1">$</option>
          <option value="2">$$</option>
          <option value="3">$$$</option>
        </select>

        <label style={styles.label}>Model:</label>
        <select value={model} onChange={(e) => setModel(e.target.value)} style={styles.select}>
          <option value="naive">Naive (KNN)</option>
          <option value="classical">Classical ML</option>
          <option value="deep">Deep Learning</option>
        </select>

        <button type="submit" style={styles.button}>Find Restaurants</button>
      </div>
    </form>
  );
}

const styles = {
  form: {
    padding: '1rem',
    display: 'flex',
    justifyContent: 'center',
    alignItems: 'center'
  },
  row: {
    display: 'flex',
    flexWrap: 'wrap',
    gap: '1rem',
    alignItems: 'center',
    justifyContent: 'center'
  },
  label: {
    fontWeight: 'bold'
  },
  select: {
    padding: '0.4rem',
    borderRadius: '4px',
    border: '1px solid #ccc'
  },
  input: {
    padding: '0.4rem',
    borderRadius: '4px',
    border: '1px solid #ccc',
    width: '160px'
  },
  button: {
    padding: '0.5rem 1rem',
    backgroundColor: '#007bff',
    color: '#fff',
    border: 'none',
    borderRadius: '4px',
    cursor: 'pointer'
  }
};

export default InputForm;
