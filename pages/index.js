import React, { useState, useRef } from 'react';
import { useSpring, animated } from '@react-spring/web';

// This is the main App component for ResumeCraft.
// It handles all UI logic, state management, and layout for the single-page application.
const App = () => {
  // State for the resume and job description text areas
  const [resumeText, setResumeText] = useState('');
  const [jobDescriptionText, setJobDescriptionText] = useState('');

  // State for UI elements and data
  const [isLoading, setIsLoading] = useState(false);
  const [results, setResults] = useState(null); // Will hold the analysis results
  const [isProUser, setIsProUser] = useState(false); // Simulates a user's subscription status

  // References for the file input and results section
  const fileInputRef = useRef(null);
  const resultsRef = useRef(null);

  // Animation for the match score display
  const scoreProps = useSpring({
    from: { number: 0 },
    to: { number: results ? results.matchScore : 0 },
    config: { duration: 1000 },
  });

  // Handle file upload (PDF/Word)
  const handleFileUpload = (e) => {
    const file = e.target.files[0];
    if (file) {
      // In a real-world app, you would parse the file on the backend.
      // Here, we just set a dummy text to simulate the process.
      const reader = new FileReader();
      reader.onload = (event) => {
        setResumeText(`
          Senior Software Engineer
          John Doe
          ...
          (The content of the uploaded file would go here)
        `);
      };
      reader.readAsText(file);
    }
  };

  // Handle drag and drop for resume upload
  const handleDragOver = (e) => {
    e.preventDefault();
  };
  const handleDrop = (e) => {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    if (file) {
      handleFileUpload({ target: { files: [file] } });
    }
  };

  // Simulate an API call to the backend for analysis
  const handleAnalyze = async () => {
    if (!resumeText || !jobDescriptionText) {
      alert('Please provide both resume and job description text.');
      return;
    }

    setIsLoading(true);
    setResults(null); // Clear previous results

    try {
      // Simulate a network delay
      await new Promise(resolve => setTimeout(resolve, 2000));

      // This is where you would call your backend API.
      // Example:
      // const response = await fetch('/api/analyze', {
      //   method: 'POST',
      //   headers: { 'Content-Type': 'application/json' },
      //   body: JSON.stringify({ resume: resumeText, jobDescription: jobDescriptionText }),
      // });
      // const data = await response.json();
      // setResults(data);

      // Using dummy data for the MVP
      const dummyResults = {
        matchScore: 67,
        missingKeywords: ['Agile', 'DevOps', 'CI/CD', 'Docker'],
        suggestedRephrases: [
          {
            original: 'Managed software development lifecycle',
            suggestion: 'Spearheaded the entire software development lifecycle, utilizing Agile methodologies.',
          },
          {
            original: 'Worked on cloud infrastructure',
            suggestion: 'Designed and implemented scalable cloud infrastructure solutions on AWS.',
          },
        ],
      };
      setResults(dummyResults);

      // Scroll to the results section after analysis
      if (resultsRef.current) {
        resultsRef.current.scrollIntoView({ behavior: 'smooth' });
      }

    } catch (error) {
      console.error('Analysis failed:', error);
      alert('Analysis failed. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-50 text-gray-800 font-sans flex flex-col items-center">
      {/* Tailwind CSS is included globally via a CDN in the HTML file, so we can use its classes directly */}
      
      {/* Header */}
      <header className="w-full bg-white shadow-md p-4 sticky top-0 z-50">
        <div className="max-w-7xl mx-auto flex items-center justify-between">
          <div className="flex items-center space-x-2">
            {/* Logo placeholder - replace with your own SVG/icon */}
            <svg xmlns="http://www.w3.org/2000/svg" className="h-8 w-8 text-green-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
            </svg>
            <h1 className="text-2xl font-bold text-gray-900">ResumeCraft</h1>
          </div>
          {/* Share button (Optional MVP Extra) */}
          <button
            onClick={() => alert('Share functionality coming soon!')}
            className="hidden sm:inline-block px-4 py-2 text-sm font-medium text-gray-600 rounded-lg hover:bg-gray-100 transition-colors"
          >
            Share
          </button>
        </div>
      </header>

      {/* Main Content Section */}
      <main className="flex-grow w-full max-w-7xl mx-auto p-4 sm:p-6 lg:p-8">
        <h2 className="text-center text-3xl sm:text-4xl font-extrabold text-gray-900 mb-8">
          Optimize Your Resume for ATS
        </h2>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
          {/* Resume Field */}
          <div className="flex flex-col space-y-4">
            <h3 className="text-xl font-semibold text-gray-700">Your Resume</h3>
            {/* Drag & Drop Area / File Upload */}
            <div
              className="border-2 border-dashed border-gray-300 rounded-xl p-6 text-center text-gray-500 hover:border-green-500 transition-colors cursor-pointer"
              onDragOver={handleDragOver}
              onDrop={handleDrop}
              onClick={() => fileInputRef.current.click()}
            >
              <svg xmlns="http://www.w3.org/2000/svg" className="mx-auto h-12 w-12 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v5" />
              </svg>
              <p className="mt-2 text-sm">Drag & drop or click to upload (PDF/DOCX)</p>
              <input type="file" ref={fileInputRef} onChange={handleFileUpload} className="hidden" accept=".pdf,.doc,.docx" />
            </div>

            {/* Paste Field */}
            <textarea
              className="w-full min-h-[300px] p-4 text-sm rounded-xl border border-gray-300 focus:ring-green-500 focus:border-green-500 transition-colors shadow-sm resize-none"
              placeholder="Or paste your resume text here..."
              value={resumeText}
              onChange={(e) => setResumeText(e.target.value)}
            />
          </div>

          {/* Job Description Field */}
          <div className="flex flex-col space-y-4">
            <h3 className="text-xl font-semibold text-gray-700">Job Description</h3>
            <textarea
              className="w-full min-h-[300px] p-4 text-sm rounded-xl border border-gray-300 focus:ring-green-500 focus:border-green-500 transition-colors shadow-sm resize-none"
              placeholder="Paste the job description here..."
              value={jobDescriptionText}
              onChange={(e) => setJobDescriptionText(e.target.value)}
            />
          </div>
        </div>

        {/* Analyze Button */}
        <div className="flex justify-center mb-8">
          <button
            onClick={handleAnalyze}
            disabled={isLoading}
            className={`px-8 py-4 text-lg font-bold rounded-full transition-all duration-300 transform hover:scale-105 shadow-lg
              ${isLoading ? 'bg-gray-400 text-gray-600 cursor-not-allowed' : 'bg-green-600 text-white hover:bg-green-700'}`}
          >
            {isLoading ? (
              <svg className="animate-spin h-5 w-5 text-white inline-block mr-2" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
              </svg>
            ) : null}
            {isLoading ? 'Analyzing...' : 'Analyze Resume'}
          </button>
        </div>

        {/* Results Section */}
        {results && (
          <animated.section
            ref={resultsRef}
            className="bg-white p-6 sm:p-8 rounded-2xl shadow-xl space-y-6 lg:space-y-8"
            style={{ opacity: results ? 1 : 0, transform: results ? 'translateY(0)' : 'translateY(20px)' }}
          >
            <h3 className="text-2xl font-bold text-gray-800">Analysis Results</h3>
            
            {/* Match Score */}
            <div>
              <p className="text-xl font-semibold text-gray-700 mb-2">Match Score</p>
              <div className="flex items-center justify-center">
                <animated.div className="text-6xl sm:text-7xl font-extrabold text-green-600 transition-colors">
                  {scoreProps.number.to(n => `${Math.round(n)}%`)}
                </animated.div>
              </div>
            </div>

            {/* Missing Keywords */}
            <div>
              <h4 className="text-lg font-semibold text-gray-700 mb-2">Missing Keywords</h4>
              <div className="flex flex-wrap gap-2">
                {results.missingKeywords.length > 0 ? (
                  results.missingKeywords.map((keyword, index) => (
                    <span key={index} className="bg-red-100 text-red-700 px-3 py-1 text-sm font-medium rounded-full">
                      {keyword}
                    </span>
                  ))
                ) : (
                  <p className="text-gray-500 text-sm">Great! You have all the key terms.</p>
                )}
              </div>
            </div>

            {/* Suggested Rephrases */}
            <div>
              <h4 className="text-lg font-semibold text-gray-700 mb-2">Suggested Rephrases</h4>
              <div className="space-y-4">
                {results.suggestedRephrases.map((item, index) => (
                  <div key={index} className="bg-gray-50 p-4 rounded-xl border border-gray-200">
                    <p className="text-sm text-gray-500 mb-1">
                      <span className="font-medium text-gray-600">Original:</span> {item.original}
                    </p>
                    <p className="text-base text-gray-700">
                      <span className="font-medium text-green-600">Suggestion:</span> {item.suggestion}
                    </p>
                    {/* Optional "Replace" button */}
                    <button
                      onClick={() => alert('Replace functionality for Pro users only.')}
                      className="mt-2 px-3 py-1 text-xs text-green-600 font-medium rounded-lg border border-green-600 hover:bg-green-50 transition-colors"
                      title="Pro feature"
                    >
                      Replace
                    </button>
                  </div>
                ))}
              </div>
            </div>

            {/* Download Optimized Resume Button */}
            <div className="text-center mt-6">
              <button
                onClick={() => alert('This is a paid feature.')}
                className={`px-6 py-3 font-bold rounded-full transition-colors duration-300
                  ${isProUser
                    ? 'bg-green-600 text-white hover:bg-green-700'
                    : 'bg-gray-200 text-gray-500 cursor-not-allowed'
                  }`}
                disabled={!isProUser}
              >
                Download Optimized Resume
                <span className="ml-2 px-2 py-0.5 text-xs bg-yellow-400 text-yellow-800 rounded-full font-bold">PRO</span>
              </button>
            </div>
          </animated.section>
        )}
      </main>

      {/* Footer */}
      <footer className="w-full bg-white shadow-inner p-4 mt-8">
        <div className="max-w-7xl mx-auto text-center text-sm text-gray-500">
          © {new Date().getFullYear()} ResumeCraft. All rights reserved.
        </div>
      </footer>
    </div>
  );
};

export default App;
