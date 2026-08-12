import { useState } from "react";
import UploadPanel from "./components/UploadPanel.jsx";
import ResultCard from "./components/ResultCard.jsx";
import { classifyImage } from "./api.js";

export default function App() {
  const [file, setFile] = useState(null);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [uploadKey, setUploadKey] = useState(0);

  const handleFileSelected = (selectedFile, validationError) => {
    setResult(null);
    setError(validationError);
    setFile(selectedFile);
  };

  const handleAnalyze = async () => {
    if (!file) return;
    setIsLoading(true);
    setError(null);
    setResult(null);
    try {
      const prediction = await classifyImage(file);
      setResult(prediction);
    } catch (err) {
      setError(err.message);
    } finally {
      setIsLoading(false);
    }
  };

  const handleClear = () => {
    setFile(null);
    setResult(null);
    setError(null);
    setUploadKey((key) => key + 1); // remounts UploadPanel, clearing its preview + file input
  };

  return (
    <div className="app">
      <header className="app-header">
        <h1>Solar Panel Condition Classifier</h1>
        <p>Upload a photo of a solar panel to detect dirt, damage, or obstructions.</p>
      </header>

      <main className="app-main">
        <UploadPanel key={uploadKey} onFileSelected={handleFileSelected} disabled={isLoading} />

        <div className="button-row">
          <button className="analyze-button" onClick={handleAnalyze} disabled={!file || isLoading}>
            {isLoading ? "Analyzing..." : "Analyze photo"}
          </button>
          {file && (
            <button className="clear-button" onClick={handleClear} disabled={isLoading}>
              Clear
            </button>
          )}
        </div>

        {error && <p className="error-message">{error}</p>}
        {result && <ResultCard result={result} />}
      </main>
    </div>
  );
}
