import { Routes, Route } from "react-router-dom";
import Home from "./pages/Home";
import Gallery from "./pages/Gallery";
import Hi from "./pages/Hi";
import HiEn from "./pages/HiEn";
import HiZh from "./pages/HiZh";
import "@/i18n/config";

export default function App() {
  return (
    <Routes>
      <Route path="/" element={<Home />} />
      <Route path="/gallery" element={<Gallery />} />
      <Route path="/hi" element={<Hi />} />
      <Route path="/hi/en" element={<HiEn />} />
      <Route path="/hi/zh" element={<HiZh />} />
    </Routes>
  );
}
