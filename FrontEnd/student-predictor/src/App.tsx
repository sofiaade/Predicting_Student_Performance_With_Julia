import { useState } from "react";

// API Configuration
const API_BASE_URL = "http://localhost:8000";

interface Metrics {
  accuracy?: number;
  precision?: number;
  recall?: number;
  f1_score?: number;
  cv_mean?: number;
  cv_std?: number;
  mse?: number;
  rmse?: number;
  mae?: number;
  r2_score?: number;
  feature_importance?: Record<string, number>;
}

interface AtRiskStudent {
  student_id: string;
  risk_level: string;
  attendance: string | number;
  test_score: string | number;
  homework: string | number;
  action: string;
}

interface Results {
  status: string;
  metrics: Metrics;
  at_risk_students: AtRiskStudent[];
  total_students: number;
  at_risk_count: number;
  plots?: {
    confusion_matrix?: string;
    distribution?: string;
    scatter?: string;
  };
}

function App() {
  // State management
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [selectedModel, setSelectedModel] = useState("logistic");
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [isTraining, setIsTraining] = useState(false);
  const [uploadStatus, setUploadStatus] = useState("");
  const [results, setResults] = useState<Results | null>(null);

  // Handle file selection
  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setSelectedFile(e.target.files[0]);
      setUploadStatus(`Selected: ${e.target.files[0].name}`);
      // Reset session when new file selected
      setSessionId(null);
      setResults(null);
    }
  };

  // Handle model selection
  const handleModelChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setSelectedModel(e.target.value);
  };

  // Upload file to backend
  const handleUpload = async () => {
    if (!selectedFile) {
      alert("Please select a CSV file first!");
      return;
    }

    setIsUploading(true);
    setUploadStatus("Uploading and preprocessing data...");

    const formData = new FormData();
    formData.append("datafile", selectedFile);

    try {
      const response = await fetch(`${API_BASE_URL}/api/upload`, {
        method: "POST",
        body: formData,
      });

      const data = await response.json();

      if (data.status === "success") {
        setSessionId(data.session_id);
        setUploadStatus(`✅ Upload successful! ${data.rows} students loaded.`);
        console.log("Session ID:", data.session_id);
      } else {
        setUploadStatus(`❌ Error: ${data.message}`);
        alert(`Upload failed: ${data.message}`);
      }
    } catch (error) {
      setUploadStatus(`❌ Error: ${error}`);
      console.error("Upload error:", error);
      alert(`Upload failed: ${error}`);
    } finally {
      setIsUploading(false);
    }
  };

  // Train model
  const handleTrain = async () => {
    if (!sessionId) {
      alert("Please upload a file first!");
      return;
    }

    setIsTraining(true);

    try {
      const response = await fetch(`${API_BASE_URL}/api/train`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ 
          model_type: selectedModel,
          session_id: sessionId 
        }),
      });

      const data = await response.json();

      if (data.status === "success") {
        setResults(data);
        alert("✅ Model trained successfully! Scroll down to see results.");
        
        // Scroll to results
        document.getElementById('results')?.scrollIntoView({ behavior: 'smooth' });
      } else {
        alert(`❌ Training error: ${data.message}`);
      }
    } catch (error) {
      alert(`❌ Error: ${error}`);
      console.error("Training error:", error);
    } finally {
      setIsTraining(false);
    }
  };

  // Combined upload and train
  const handleUploadAndTrain = async () => {
    if (!selectedFile) {
      alert("Please select a CSV file first!");
      return;
    }

    // If not uploaded yet, upload first
    if (!sessionId) {
      await handleUpload();
    }
    
    // Small delay to ensure state updates
    setTimeout(() => {
      if (sessionId) {
        handleTrain();
      }
    }, 500);
  };

  // Get risk badge color
  const getRiskBadgeClass = (riskLevel: string) => {
    if (riskLevel.includes("High")) return "risk-high";
    if (riskLevel.includes("Medium")) return "risk-medium";
    return "risk-low";
  };

  return (
    <>
      {/* Navigation */}
      <nav className="navbar">
        <div className="nav-container">
          <div className="logo">📚 Mini Bytes Student Predictor</div>
          <ul className="nav-links">
            <li>
              <a href="#home">Home</a>
            </li>
            <li>
              <a href="#upload">Upload Data</a>
            </li>
            <li>
              <a href="#results">Results</a>
            </li>
            <li>
              <a href="#about">About</a>
            </li>
          </ul>
        </div>
      </nav>

      {/* Hero Section */}
      <section className="hero" id="home">
        <h1>🎓 Predict Student Performance with AI</h1>
        <p>
          Identify at-risk students early and provide timely interventions using
          machine learning
        </p>
        <div className="cta-buttons">
          <a href="#upload" className="btn btn-primary">
            Get Started →
          </a>
          <a href="#results" className="btn btn-secondary">
            View Results
          </a>
        </div>
      </section>

      {/* Features Section */}
      <section className="features">
        <div className="container">
          <h2 className="section-title">How It Works</h2>
          <div className="feature-grid">
            <div className="feature-card">
              <div className="icon">📊</div>
              <h3>Upload Your Data</h3>
              <p>
                Simply upload a CSV file with student attendance, grades,
                homework completion, and other metrics
              </p>
            </div>
            <div className="feature-card">
              <div className="icon">🤖</div>
              <h3>Choose Your Model</h3>
              <p>
                Select from logistic regression, decision trees, or linear
                regression based on your needs
              </p>
            </div>
            <div className="feature-card">
              <div className="icon">📈</div>
              <h3>Get Insights</h3>
              <p>
                View predictions, visualizations, and actionable recommendations
                for at-risk students
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* Stats Banner */}
      <div className="stats-banner">
        <div className="container">
          <div className="stats-grid">
            <div className="stat-item">
              <h3>{results?.metrics?.accuracy?.toFixed(1) || '94'}%</h3>
              <p>Prediction Accuracy</p>
            </div>
            <div className="stat-item">
              <h3>3</h3>
              <p>ML Models Available</p>
            </div>
            <div className="stat-item">
              <h3>{results?.total_students || 'Fast'}</h3>
              <p>{results?.total_students ? 'Students Analyzed' : 'Real-time Analysis'}</p>
            </div>
          </div>
        </div>
      </div>

      {/* Upload Section */}
      <section className="upload-section" id="upload">
        <div className="container">
          <h2 className="section-title">Upload Student Data</h2>
          <div className="upload-container">
            <h3>📁 Select Your CSV File</h3>
            <p style={{ color: "#666", margin: "1rem 0" }}>
              Upload student data including attendance, test scores, homework
              completion, etc.
            </p>
            <div className="file-upload-area">
              <label className="file-label">
                <input 
                  type="file" 
                  accept=".csv" 
                  onChange={handleFileChange}
                  disabled={isUploading || isTraining}
                />
                📤 Choose File
              </label>
              <p style={{ marginTop: "1rem", color: uploadStatus.includes("✅") ? "#28a745" : "#999", fontSize: "0.9rem" }}>
                {uploadStatus || "Supported format: CSV"}
              </p>
              {sessionId && (
                <p style={{ marginTop: "0.5rem", color: "#28a745", fontSize: "0.85rem" }}>
                  ✓ Session active: {sessionId.slice(0, 8)}...
                </p>
              )}
            </div>

            <div className="model-selection">
              <h3>Select Machine Learning Model</h3>
              <div className="radio-group">
                <label className="radio-item">
                  <input
                    type="radio"
                    name="model"
                    value="logistic"
                    checked={selectedModel === "logistic"}
                    onChange={handleModelChange}
                    disabled={isTraining}
                  />
                  <div className="radio-item-content">
                    <strong>Logistic Regression</strong>
                    <small>
                      Best for binary classification (pass/fail predictions)
                    </small>
                  </div>
                </label>
                <label className="radio-item">
                  <input 
                    type="radio" 
                    name="model" 
                    value="decision_tree"
                    checked={selectedModel === "decision_tree"}
                    onChange={handleModelChange}
                    disabled={isTraining}
                  />
                  <div className="radio-item-content">
                    <strong>Decision Tree</strong>
                    <small>Interpretable model showing decision paths</small>
                  </div>
                </label>
                <label className="radio-item">
                  <input 
                    type="radio" 
                    name="model" 
                    value="linear"
                    checked={selectedModel === "linear"}
                    onChange={handleModelChange}
                    disabled={isTraining}
                  />
                  <div className="radio-item-content">
                    <strong>Linear Regression</strong>
                    <small>
                      Best for predicting continuous scores (GPA, test scores)
                    </small>
                  </div>
                </label>
              </div>
            </div>

            <button 
              className="btn btn-primary" 
              style={{ marginTop: "2rem" }}
              onClick={sessionId ? handleTrain : handleUploadAndTrain}
              disabled={!selectedFile || isUploading || isTraining}
            >
              {isUploading ? "⏳ Uploading..." : isTraining ? "⏳ Training..." : "🚀 Train Model & Predict"}
            </button>
          </div>
        </div>
      </section>

      {/* Results Section */}
      <section className="results-section" id="results">
        <div className="container">
          <h2 className="section-title">Prediction Results</h2>
          
          {!results ? (
            <div style={{ textAlign: 'center', padding: '3rem', color: '#666' }}>
              <p style={{ fontSize: '1.2rem' }}>📊 No results yet</p>
              <p>Upload data and train a model to see predictions</p>
            </div>
          ) : (
            <>
              <div className="metrics-grid">
                {results.metrics.accuracy !== undefined && (
                  <div className="metric-card">
                    <div className="metric-label">Accuracy</div>
                    <div className="metric-value">{results.metrics.accuracy.toFixed(1)}%</div>
                    {results.metrics.cv_mean && (
                      <small style={{ color: '#666', fontSize: '0.85rem' }}>
                        CV: {results.metrics.cv_mean.toFixed(1)}% (±{results.metrics.cv_std?.toFixed(1)}%)
                      </small>
                    )}
                  </div>
                )}
                {results.metrics.precision !== undefined && (
                  <div className="metric-card">
                    <div className="metric-label">Precision</div>
                    <div className="metric-value">{results.metrics.precision.toFixed(1)}%</div>
                  </div>
                )}
                {results.metrics.recall !== undefined && (
                  <div className="metric-card">
                    <div className="metric-label">Recall</div>
                    <div className="metric-value">{results.metrics.recall.toFixed(1)}%</div>
                  </div>
                )}
                {results.metrics.f1_score !== undefined && (
                  <div className="metric-card">
                    <div className="metric-label">F1 Score</div>
                    <div className="metric-value">{results.metrics.f1_score.toFixed(1)}%</div>
                  </div>
                )}
                {results.metrics.r2_score !== undefined && (
                  <div className="metric-card">
                    <div className="metric-label">R² Score</div>
                    <div className="metric-value">{results.metrics.r2_score.toFixed(1)}%</div>
                  </div>
                )}
                {results.metrics.rmse !== undefined && (
                  <div className="metric-card">
                    <div className="metric-label">RMSE</div>
                    <div className="metric-value">{results.metrics.rmse.toFixed(2)}</div>
                  </div>
                )}
              </div>

              {/* Feature Importance for Decision Tree */}
              {results.metrics.feature_importance && (
                <div className="visualization-area" style={{ marginTop: '2rem' }}>
                  <h3 style={{ marginBottom: "1.5rem" }}>
                    🔍 Most Important Features
                  </h3>
                  <div style={{ background: 'white', padding: '1.5rem', borderRadius: '8px', boxShadow: '0 2px 8px rgba(0,0,0,0.1)' }}>
                    {Object.entries(results.metrics.feature_importance)
                      .slice(0, 5)
                      .map(([feature, importance], idx) => (
                        <div key={feature} style={{ marginBottom: '1rem' }}>
                          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.5rem' }}>
                            <span style={{ fontWeight: 500 }}>{idx + 1}. {feature}</span>
                            <span style={{ color: '#666' }}>{(importance * 100).toFixed(1)}%</span>
                          </div>
                          <div style={{ background: '#e0e0e0', height: '8px', borderRadius: '4px', overflow: 'hidden' }}>
                            <div style={{ 
                              background: 'linear-gradient(90deg, #667eea 0%, #764ba2 100%)', 
                              height: '100%', 
                              width: `${importance * 100}%`,
                              transition: 'width 0.3s ease'
                            }} />
                          </div>
                        </div>
                      ))}
                  </div>
                </div>
              )}

              {/* Visualizations */}
              {results.plots && (
                <div className="visualization-area" style={{ marginTop: '2rem' }}>
                  <h3 style={{ marginBottom: "1.5rem" }}>
                    📊 Performance Visualization
                  </h3>
                  <div style={{ display: 'grid', gap: '2rem', gridTemplateColumns: 'repeat(auto-fit, minmax(400px, 1fr))' }}>
                    {results.plots.confusion_matrix && (
                      <div style={{ background: 'white', padding: '1rem', borderRadius: '8px', boxShadow: '0 2px 8px rgba(0,0,0,0.1)' }}>
                        <h4 style={{ marginBottom: '1rem' }}>Confusion Matrix</h4>
                        <img 
                          src={`data:image/png;base64,${results.plots.confusion_matrix}`}
                          alt="Confusion Matrix"
                          style={{ maxWidth: '100%', height: 'auto' }}
                        />
                      </div>
                    )}
                    {results.plots.distribution && (
                      <div style={{ background: 'white', padding: '1rem', borderRadius: '8px', boxShadow: '0 2px 8px rgba(0,0,0,0.1)' }}>
                        <h4 style={{ marginBottom: '1rem' }}>Distribution</h4>
                        <img 
                          src={`data:image/png;base64,${results.plots.distribution}`}
                          alt="Distribution"
                          style={{ maxWidth: '100%', height: 'auto' }}
                        />
                      </div>
                    )}
                    {results.plots.scatter && (
                      <div style={{ background: 'white', padding: '1rem', borderRadius: '8px', boxShadow: '0 2px 8px rgba(0,0,0,0.1)' }}>
                        <h4 style={{ marginBottom: '1rem' }}>Predicted vs Actual</h4>
                        <img 
                          src={`data:image/png;base64,${results.plots.scatter}`}
                          alt="Scatter Plot"
                          style={{ maxWidth: '100%', height: 'auto' }}
                        />
                      </div>
                    )}
                  </div>
                </div>
              )}

              {/* At-Risk Students Table */}
              <div className="at-risk-table" style={{ marginTop: '3rem' }}>
                <h3 style={{ marginBottom: "1.5rem" }}>
                  ⚠️ At-Risk Students ({results.at_risk_count} identified)
                </h3>
                {results.at_risk_students.length > 0 ? (
                  <>
                    <table>
                      <thead>
                        <tr>
                          <th>Student ID</th>
                          <th>Risk Level</th>
                          <th>Attendance</th>
                          <th>Test Score</th>
                          <th>Homework</th>
                          <th>Action</th>
                        </tr>
                      </thead>
                      <tbody>
                        {results.at_risk_students.map((student) => (
                          <tr key={student.student_id}>
                            <td>{student.student_id}</td>
                            <td>
                              <span className={`risk-badge ${getRiskBadgeClass(student.risk_level)}`}>
                                {student.risk_level}
                              </span>
                            </td>
                            <td>{student.attendance}</td>
                            <td>{student.test_score}</td>
                            <td>{student.homework}</td>
                            <td>{student.action}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                    <div style={{ marginTop: "2rem", textAlign: "center" }}>
                      <button 
                        className="btn btn-primary"
                        onClick={() => {
                          const dataStr = JSON.stringify(results, null, 2);
                          const dataBlob = new Blob([dataStr], { type: 'application/json' });
                          const url = URL.createObjectURL(dataBlob);
                          const link = document.createElement('a');
                          link.href = url;
                          link.download = `student-predictions-${new Date().toISOString().split('T')[0]}.json`;
                          link.click();
                          URL.revokeObjectURL(url);
                        }}
                      >
                        📥 Download Full Report
                      </button>
                    </div>
                  </>
                ) : (
                  <p style={{ textAlign: 'center', color: '#28a745', padding: '2rem' }}>
                    ✅ Great news! No students are currently at risk.
                  </p>
                )}
              </div>
            </>
          )}
        </div>
      </section>

      {/* Footer */}
      <footer>
        <div className="container">
          <p>
            <strong>Mini Bytes Team</strong> | CMPS 4010 - Survey of Programming
            Languages
          </p>
          <p style={{ marginTop: "0.5rem", opacity: "0.8" }}>
            Sofiat Adeyemi • Shakurah Watson • Areeba Ahmad • Terri Crawford
          </p>
          <p style={{ marginTop: "1rem", fontSize: "0.9rem" }}>
            &copy; 2024 All Rights Reserved
          </p>
        </div>
      </footer>
    </>
  );
}

export default App;