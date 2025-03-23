import { useState } from "react";
import { Link } from "react-router-dom";

export default function Report() {
  const [modalOpen, setModalOpen] = useState(false);
  const [file, setFile] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [selectedModel, setSelectedModel] = useState(null);

  // Define available models with images
  const models = [
    { name: "Brest Cancer", value: "brest", image: "/images/brest.jpg" },
    { name: "Diabetes Risk Analyzer", value: "diabetes", image: "/images/diabetes.jpg" },
    { name: "Heart Cancer", value: "heart", image: "/images/heart.jpg" },
    { name: "Cancer Risk", value: "cancer", image: "/images/cancer.jpg" },
    { name: "PSAD", value: "psad", image: "/images/psad.jpg" },
    { name: "Text to Speech Converter", value: "text", image: "/images/text.jpg" },
    { name: "Blood Report Analyzer", value: "blood", image: "/images/blood.jpg" },
  ];

  const handleFileChange = (event) => {
    const selectedFile = event.target.files[0];
    if (selectedFile) {
      setFile(selectedFile);
      setPrediction(null);
    }
  };

  const handlePredict = async () => {
    if (!file || !selectedModel) {
      alert("Please select a file and a model!");
      return;
    }

    const formData = new FormData();
    formData.append("file", file);
    formData.append("model_name", selectedModel);

    setLoading(true);

    try {
      const response = await fetch("http://127.0.0.1:9000/predict/", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error("Prediction failed");
      }

      const data = await response.json();
      console.log("API Response:", data);
      setPrediction(`Prediction: ${data.prediction}`);
    } catch (error) {
      console.error("Error:", error);
      alert("Error processing the image");
    } finally {
      setLoading(false);
    }
  };
// ✅ Reset file and close modal
  const handleCloseModal = () => {
  setModalOpen(false);
  setFile(null);        // Reset the uploaded file
  setPrediction(null);  // Reset prediction
  };
  return (
    <div className="min-h-screen bg-gradient-to-br from-black to-gray-900 p-8 flex flex-col items-center justify-center relative">
      {/* Top Left Corner - Logo + Dragon */}
      <Link to="/">
        <div className="absolute top-6 left-8 flex items-center cursor-pointer">
          <img src="/images/logo1.jpg" alt="logo" className="h-10 w-10" />
          <span className="text-2xl font-bold text-white">ragon.</span>
        </div>
      </Link>

      {/* Home Button with White Border */}
      <Link
        to="/"
        className="absolute top-6 right-6 text-white px-6 py-2 rounded-lg text-lg font-semibold shadow-md border border-white 
        hover:bg-white hover:text-black transition duration-300"
      >
        Home
      </Link>

      {/* Page Heading with Glow Effect */}
      <h1 className="text-5xl font-semibold text-white text-center mb-10 tracking-tight 
        drop-shadow-[0_0_20px_rgba(255,255,255,0.7)] animate-pulse">
        Report Models
      </h1>

      {/* Model Selection Cards with Lifting Hover Effect */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8">
        {models.map((model) => (
          <button
            key={model.value}
            onClick={() => {
              setSelectedModel(model.value);
              setModalOpen(true);
            }}
            className="bg-gray-900 rounded-xl shadow-md p-4 flex flex-col items-center justify-center
            w-52 h-60 border border-gray-700 transition-transform duration-300 hover:-translate-y-2"
          >
            <img src={model.image} alt={model.name} className="w-32 h-32 rounded-lg mb-3 shadow-md" />
            <p className="text-white text-xl font-semibold">{model.name}</p>
          </button>
        ))}
      </div>

      {/* Modal for File Upload */}
      {modalOpen && (
        <div className="fixed inset-0 flex items-center justify-center bg-black bg-opacity-50">
          <div className="bg-gray-800 p-6 rounded-xl shadow-2xl w-[450px] text-white">
            <h2 className="text-2xl font-bold mb-6 text-center">
              Upload Image for {models.find((m) => m.value === selectedModel)?.name}
            </h2>

            {/* File Upload Box */}
            <div className="border-dashed border-2 border-gray-500 rounded-lg p-6 text-center bg-gray-900">
            <p className="text-gray-400 text-lg">Drag and Drop</p>
            <p className="text-gray-500 text-sm">or</p>

            {/* Smaller & Lower Browse Button */}
            <label 
              htmlFor="file-upload" 
              className="cursor-pointer bg-purple-600 text-white px-3 py-0.5 rounded text-md hover:bg-purple-500 transition mt-4 inline-block"
            >
              Browse file
            </label>
            <input id="file-upload" type="file" className="hidden" onChange={handleFileChange} />

              {/* Show selected file name */}
              {file && <p className="mt-3 text-gray-300 text-lg">Selected file: {file.name}</p>}
            </div>
            {/* Buttons */}
            <div className="flex justify-end mt-6">
              <button
                className="px-4 py-2 bg-red-500 text-white rounded-lg text-lg mr-2 hover:bg-red-400 transition"
                onClick={handleCloseModal}
              >
                Close
              </button>
              <button
                className="px-4 py-2 bg-green-500 text-white rounded-lg text-lg hover:bg-green-400 transition"
                onClick={handlePredict}
                disabled={loading}
              >
                {loading ? "Processing..." : "Predict"}
              </button>
            </div>

            {/* Prediction Result */}
            {prediction && (
              <div className="mt-4 p-4 bg-gray-700 rounded-lg">
                <h3 className="text-lg font-semibold">Prediction Result:</h3>
                <p className="text-gray-300 text-lg">{prediction}</p>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
