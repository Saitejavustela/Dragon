import Dashboard from './components/Dashboard';
import Image from './components/Image';
import Report from './components/Report';
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
export default function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<Dashboard />} />
        <Route path="/images" element={<Image />} />
        <Route path="/reports" element={<Report />} />
      </Routes>
    </Router>
  );
}