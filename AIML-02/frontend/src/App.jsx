import { useEffect, useState } from "react";
import "./App.css";

const API = "";
const DEMO_DOCTOR_ID = 2;

async function apiRequest(url, options = {}) {
  const response = await fetch(`${API}${url}`, options);
  const result = await response.json();

  if (!response.ok) {
    throw new Error(result.detail || result.message || "Request failed");
  }

  return result;
}

function App() {
  const [activeTab, setActiveTab] = useState("upload");
  const [file, setFile] = useState(null);
  const [explain, setExplain] = useState(true);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState(null);

  const [reviews, setReviews] = useState([]);
  const [selectedScan, setSelectedScan] = useState(null);
  const [history, setHistory] = useState([]);
  const [comments, setComments] = useState("");
  const [decision, setDecision] = useState("needs_review");
  const [reviewLoading, setReviewLoading] = useState(false);

  const [doctorId, setDoctorId] = useState(DEMO_DOCTOR_ID);

  async function uploadImage(event) {
    event.preventDefault();

    if (!file) {
      setError("Please select an X-ray image.");
      return;
    }

    setLoading(true);
    setError("");
    setResult(null);

    try {
      const form = new FormData();
      form.append("file", file);
      form.append("explain", String(explain));

      const response = await apiRequest("/api/v1/predict", {
        method: "POST",
        body: form,
      });

      setResult(response.data);
      setFile(null);
      event.target.reset();
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  async function loadReviews() {
    setLoading(true);
    setError("");

    try {
      const response = await apiRequest(
        `/api/v1/doctor/reviews/pending?doctor_id=${doctorId}`
      );

      setReviews(response.reviews || []);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  async function openScan(scan) {
    setSelectedScan(scan);
    setHistory([]);
    setComments("");
    setDecision("needs_review");
    setError("");

    try {
      const response = await apiRequest(
        `/api/v1/doctor/scans/${scan.scan_id}?doctor_id=${doctorId}`
      );

      setSelectedScan({
        ...scan,
        details: response,
      });

      const historyResponse = await apiRequest(
        `/api/v1/doctor/scans/${scan.scan_id}/history?doctor_id=${doctorId}`
      );

      setHistory(historyResponse.history || []);
    } catch (err) {
      setError(err.message);
    }
  }

  async function submitReview(event) {
    event.preventDefault();

    if (!selectedScan) return;

    setReviewLoading(true);
    setError("");

    try {
      await apiRequest(
        `/api/v1/doctor/scans/${selectedScan.scan_id}/review?doctor_id=${doctorId}`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            decision,
            comments,
          }),
        }
      );

      await openScan(selectedScan);
      await loadReviews();
      setComments("");
    } catch (err) {
      setError(err.message);
    } finally {
      setReviewLoading(false);
    }
  }

  useEffect(() => {
    if (activeTab === "reviews") {
      loadReviews();
    }
  }, [activeTab]);

  const imageUrl = result?.image;
  const explanation = result?.explanation_urls || {};

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand">
          <div className="brand-icon">M</div>
          <div>
            <h1>MedVision AI</h1>
            <span>Medical Imaging Research Platform</span>
          </div>
        </div>

        <div className="topbar-right">
          <span className="status-dot"></span>
          Research Prototype
        </div>
      </header>

      <main className="main-container">
        <section className="hero">
          <div>
            <p className="eyebrow">AI-ASSISTED IMAGING WORKFLOW</p>
            <h2>Medical Image Analysis</h2>
            <p>
              Upload chest X-rays, explore experimental AI predictions,
              and manage doctor-review workflows.
            </p>
          </div>

          <div className="hero-badge">
            <span>GPU</span>
            <strong>AI Inference</strong>
          </div>
        </section>

        <nav className="tabs">
          <button
            className={activeTab === "upload" ? "tab active" : "tab"}
            onClick={() => {
              setActiveTab("upload");
              setError("");
            }}
          >
            Image Analysis
          </button>

          <button
            className={activeTab === "reviews" ? "tab active" : "tab"}
            onClick={() => {
              setActiveTab("reviews");
              setError("");
            }}
          >
            Doctor Review Queue
          </button>
        </nav>

        {error && (
          <div className="error-banner">
            <strong>Error:</strong> {error}
            <button onClick={() => setError("")}>×</button>
          </div>
        )}

        {activeTab === "upload" && (
          <section className="content-grid">
            <div className="card upload-card">
              <div className="card-heading">
                <div>
                  <p className="eyebrow">STEP 01</p>
                  <h3>Upload Chest X-ray</h3>
                </div>
                <span className="number-badge">01</span>
              </div>

              <form onSubmit={uploadImage}>
                <label className="dropzone">
                  <input
                    type="file"
                    accept=".jpg,.jpeg,.png,image/jpeg,image/png"
                    onChange={(event) => setFile(event.target.files[0])}
                  />

                  <div className="upload-icon">↑</div>
                  <strong>
                    {file ? file.name : "Choose an X-ray image"}
                  </strong>
                  <span>JPEG or PNG · Maximum 15 MB</span>
                </label>

                <label className="checkbox-row">
                  <input
                    type="checkbox"
                    checked={explain}
                    onChange={(event) => setExplain(event.target.checked)}
                  />
                  Generate Grad-CAM explanation
                </label>

                <button className="primary-button" disabled={loading}>
                  {loading ? "Analyzing image..." : "Analyze X-ray"}
                </button>
              </form>
            </div>

            <div className="card workflow-card">
              <p className="eyebrow">WORKFLOW</p>
              <h3>Analysis Pipeline</h3>

              <div className="workflow-step">
                <span>01</span>
                <div>
                  <strong>Image Upload</strong>
                  <small>Validated JPEG and PNG images</small>
                </div>
              </div>

              <div className="workflow-step">
                <span>02</span>
                <div>
                  <strong>AI Inference</strong>
                  <small>PyTorch chest X-ray classifier</small>
                </div>
              </div>

              <div className="workflow-step">
                <span>03</span>
                <div>
                  <strong>Explainability</strong>
                  <small>Grad-CAM heatmap generation</small>
                </div>
              </div>

              <div className="workflow-step">
                <span>04</span>
                <div>
                  <strong>Doctor Review</strong>
                  <small>Separate review and audit records</small>
                </div>
              </div>
            </div>

            {result && (
              <>
                <div className="card result-card full-width">
                  <div className="card-heading">
                    <div>
                      <p className="eyebrow">ANALYSIS RESULT</p>
                      <h3>AI Prediction</h3>
                    </div>
                    <span className="result-label">Research output</span>
                  </div>

                  <div className="prediction-summary">
                    <div>
                      <small>Predicted class</small>
                      <h2
                        className={
                          result.prediction === "PNEUMONIA"
                            ? "prediction-positive"
                            : "prediction-normal"
                        }
                      >
                        {result.prediction}
                      </h2>
                    </div>

                    <div>
                      <small>Highest class score</small>
                      <h2>
                        {(
                          Math.max(
                            ...Object.values(result.probabilities || {})
                          ) * 100
                        ).toFixed(2)}
                        %
                      </h2>
                    </div>

                    <div>
                      <small>Model epoch</small>
                      <h2>{result.model_epoch}</h2>
                    </div>
                  </div>

                  <h4>Class scores</h4>

                  {Object.entries(result.probabilities || {}).map(
                    ([name, value]) => (
                      <div className="probability-row" key={name}>
                        <div>
                          <span>{name}</span>
                          <strong>{(value * 100).toFixed(2)}%</strong>
                        </div>
                        <div className="progress-track">
                          <div
                            className="progress-fill"
                            style={{ width: `${value * 100}%` }}
                          />
                        </div>
                      </div>
                    )
                  )}

                  <div className="metadata-grid">
                    <div>
                      <small>Request ID</small>
                      <span>{result.request_id}</span>
                    </div>
                    <div>
                      <small>Database scan ID</small>
                      <span>
                        {result.database_records?.scan_id ?? "Unavailable"}
                      </span>
                    </div>
                    <div>
                      <small>Database status</small>
                      <span>
                        {result.database_records?.database_status ?? "Unknown"}
                      </span>
                    </div>
                    <div>
                      <small>Uploaded file</small>
                      <span>{result.uploaded_filename}</span>
                    </div>
                  </div>
                </div>

                <div className="card full-width">
                  <p className="eyebrow">IMAGE EXPLANATION</p>
                  <h3>Grad-CAM Visualization</h3>

                  {imageUrl && (
                    <div className="visual-grid">
                      <div>
                        <h4>Uploaded image</h4>
                        <img src={imageUrl} alt="Uploaded chest X-ray" />
                      </div>

                      {explanation.heatmap && (
                        <div>
                          <h4>Heatmap</h4>
                          <img
                            src={explanation.heatmap}
                            alt="Grad-CAM heatmap"
                          />
                        </div>
                      )}

                      {explanation.overlay && (
                        <div>
                          <h4>Overlay</h4>
                          <img
                            src={explanation.overlay}
                            alt="Grad-CAM overlay"
                          />
                        </div>
                      )}
                    </div>
                  )}

                  {!result.explanation_generated && (
                    <p className="muted">
                      Grad-CAM was not generated for this prediction.
                    </p>
                  )}
                </div>

                <div className="card warning-card full-width">
                  <strong>Important clinical limitation</strong>
                  <p>{result.clinical_warning}</p>
                  <p>
                    Model scores are not confirmed disease probabilities.
                    Grad-CAM is an experimental explanation, not clinical
                    localization.
                  </p>
                </div>
              </>
            )}
          </section>
        )}

        {activeTab === "reviews" && (
          <section className="review-layout">
            <div className="card queue-card">
              <div className="card-heading">
                <div>
                  <p className="eyebrow">DOCTOR WORKSPACE</p>
                  <h3>Pending Review Queue</h3>
                </div>

                <button
                  className="secondary-button"
                  onClick={loadReviews}
                  disabled={loading}
                >
                  Refresh
                </button>
              </div>

              <label className="doctor-id-field">
                Demo reviewer ID
                <input
                  type="number"
                  min="1"
                  value={doctorId}
                  onChange={(event) =>
                    setDoctorId(Number(event.target.value))
                  }
                />
              </label>

              <p className="demo-notice">
                Local demo reviewer only. This is not authenticated
                clinical access.
              </p>

              <div className="queue-list">
                {reviews.length === 0 && (
                  <p className="muted">
                    No pending reviews found.
                  </p>
                )}

                {reviews.map((item) => (
                  <button
                    className={
                      selectedScan?.scan_id === item.scan.scan_id
                        ? "queue-item selected"
                        : "queue-item"
                    }
                    key={item.scan.scan_id}
                    onClick={() => openScan(item.scan)}
                  >
                    <div className="queue-item-top">
                      <strong>Scan #{item.scan.scan_id}</strong>
                      <span className="pending-pill">
                        {item.review_status}
                      </span>
                    </div>

                    <span>{item.scan.original_filename}</span>
                    <span>
                      AI result: {item.prediction.predicted_class}
                    </span>
                  </button>
                ))}
              </div>
            </div>

            <div className="card review-detail-card">
              {!selectedScan && (
                <div className="empty-state">
                  <div>◫</div>
                  <h3>Select a scan</h3>
                  <p>Choose a scan from the review queue to see its details.</p>
                </div>
              )}

              {selectedScan && (
                <>
                  <p className="eyebrow">REVIEW WORKSPACE</p>
                  <h3>Scan #{selectedScan.scan_id}</h3>

                  <div className="detail-line">
                    <span>File</span>
                    <strong>{selectedScan.original_filename}</strong>
                  </div>

                  <div className="detail-line">
                    <span>Patient record</span>
                    <strong>{selectedScan.patient_id}</strong>
                  </div>

                  {selectedScan.details?.prediction && (
                    <>
                      <div className="detail-line">
                        <span>AI prediction</span>
                        <strong>
                          {selectedScan.details.prediction.predicted_class}
                        </strong>
                      </div>

                      <div className="detail-line">
                        <span>Model score</span>
                        <strong>
                          {(
                            selectedScan.details.prediction.confidence * 100
                          ).toFixed(2)}
                          %
                        </strong>
                      </div>
                    </>
                  )}

                  <h4>Submit review</h4>

                  <form onSubmit={submitReview}>
                    <label className="form-label">
                      Review decision
                      <select
                        value={decision}
                        onChange={(event) => setDecision(event.target.value)}
                      >
                        <option value="needs_review">Needs review</option>
                        <option value="approved">Approved</option>
                        <option value="rejected">Rejected</option>
                      </select>
                    </label>

                    <label className="form-label">
                      Comments
                      <textarea
                        value={comments}
                        onChange={(event) => setComments(event.target.value)}
                        placeholder="Enter review comments..."
                        maxLength={5000}
                        rows={4}
                      />
                    </label>

                    <button
                      className="primary-button"
                      disabled={reviewLoading}
                    >
                      {reviewLoading ? "Saving..." : "Submit review"}
                    </button>
                  </form>

                  <h4>Review history</h4>

                  {history.length === 0 && (
                    <p className="muted">No previous reviews.</p>
                  )}

                  {history.map((entry) => (
                    <div className="history-entry" key={entry.review_id}>
                      <div>
                        <strong>{entry.decision}</strong>
                        <small>{entry.reviewer}</small>
                      </div>
                      <p>{entry.comments || "No comments provided."}</p>
                      <small>{entry.created_at}</small>
                    </div>
                  ))}
                </>
              )}
            </div>
          </section>
        )}

        <footer className="footer">
          Medical AI Research Prototype · Not validated for clinical use
        </footer>
      </main>
    </div>
  );
}

export default App;
