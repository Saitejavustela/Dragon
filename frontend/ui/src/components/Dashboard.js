import { Image, FileText } from "lucide-react";
import { Link } from "react-router-dom";
import { ReactTyped } from "react-typed"; // Fixed import

export default function Dashboard() {
  return (
    <div className="min-h-screen bg-black p-8 flex flex-col items-center justify-center relative">
      {/* Top Left Corner - Logo + Dragon. */}
      <div className="absolute top-6 left-8 flex items-center cursor-pointer">
        <img src="/images/logo1.jpg" alt="logo" className="h-10 w-10" />
        <span className="text-2xl font-bold text-white">ragon.</span>
      </div>

      {/* Centered Content Wrapper */}
      <div className="flex flex-col items-center">
        {/* Heading */}
        <h1 className="text-6xl font-semibold text-white text-center mb-12 tracking-tight">
          <div className="drop-shadow-[0_0_6px_rgba(255,255,255,0.8)]">
            Diagnostic Report Analysis
          </div>
          {/* Background Box for Typing Effect */}
          <span className="text-5xl font-semibold text-purple-600">
            <ReactTyped
              strings={["General Optimization"]}
              typeSpeed={50}
              backSpeed={30}
              loop
            />
          </span>
        </h1>
        
        {/* Additional Content */}
        <p className="text-lg text-gray-300 text-center mb-8">
          Enhance diagnosis with DRAGON—powerful models for analyzing medical images and records, driving automation and get suggested Earlier.
        </p>

        {/* Cards Section */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-10">
          {/* Images Card */}
          <Link
            to="/images"
            className="group cursor-pointer transition-all duration-300 hover:-translate-y-3 w-full max-w-lg"
          >
            <div className="bg-gray-900 rounded-2xl shadow-xl p-10 flex flex-col items-center justify-center 
            hover:bg-purple-500 transition-colors h-64 w-80 border border-purple-800">
              
              {/* Inner Circle with Glow on Hover */}
              <div className="bg-purple-900 p-6 rounded-full mb-6 transition-all duration-300 
              group-hover:drop-shadow-[0_0_15px_rgba(128,0,128,0.8)]">
                <Image className="h-16 w-16 text-purple-400" />
              </div>
              <h2 className="text-2xl font-bold text-purple-300">Images</h2>
            </div>
          </Link>

          {/* Reports Card */}
          <Link
            to="/reports"
            className="group cursor-pointer transition-all duration-300 hover:-translate-y-3 w-full max-w-lg"
          >
            <div className="bg-gray-900 rounded-2xl shadow-xl p-10 flex flex-col items-center justify-center 
            hover:bg-blue-500 transition-colors h-64 w-80 border border-blue-800">
              
              {/* Inner Circle with Glow on Hover */}
              <div className="bg-blue-900 p-6 rounded-full mb-6 transition-all duration-300 
              group-hover:drop-shadow-[0_0_15px_rgba(0,0,255,0.8)]">
                <FileText className="h-16 w-16 text-blue-400" />
              </div>

              <h2 className="text-2xl font-bold text-blue-300">Reports</h2>
            </div>
          </Link>
        </div>
      </div>
    </div>
  );
}
