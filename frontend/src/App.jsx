import { useState, useEffect } from 'react';
import Dashboard from './components/Dashboard';
import Threads from './components/Threads';
import Particles from './components/Particles';
import GridPattern from './components/GridPattern';

function App() {
  return (
    <div className="App relative min-h-screen">
      {/* Solid Color Background */}
      <div className="fixed inset-0 bg-indigo-50 -z-10" />

      {/* Grid Pattern */}
      <GridPattern
        width={60}
        height={60}
        squares={[
          [0, 1],
          [1, 3],
          [3, 0],
          [5, 2],
          [7, 4],
          [10, 1],
          [12, 5],
          [15, 3]
        ]}
      />

      {/* Particles Effect */}
      <Particles
        quantity={30}
        color="rgba(99, 102, 241, 0.4)"
        size={3}
        speed={0.3}
      />

      {/* Threads Animation */}
      <div style={{ width: '100%', height: '100vh', position: 'fixed', top: 0, left: 0, zIndex: 1 }}>
        <Threads
          color={[0.388, 0.400, 0.945]}
          amplitude={1.2}
          distance={0}
          enableMouseInteraction={true}
        />
      </div>

      {/* Main Content */}
      <div className="relative z-10">
        <Dashboard />
      </div>
    </div>
  );
}

export default App;
