import { useState } from 'react'
import { BrowserRouter as Router, Route, Routes } from 'react-router-dom'

import './App.css'
import Main from './pages/Main.jsx';

function App() {
  const [count, setCount] = useState(0)

  return (
     <Router>
           
           
            <div className="App">
                <Routes>
                    
                    <Route path="/" element={<Main />} />
                </Routes>
            </div>
        </Router>
  )
}

export default App
