import { useState, useEffect, useRef } from "react";

const API_BASE = "http://localhost:8000";

// ─── Design tokens ────────────────────────────────────────────────────────────
const css = `
  @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Barlow:wght@300;400;500;600;700&family=Barlow+Condensed:wght@400;600;700&display=swap');

  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  :root {
    --bg-void:     #090c0f;
    --bg-deep:     #0d1117;
    --bg-panel:    #111820;
    --bg-card:     #161e28;
    --bg-hover:    #1c2736;
    --border:      #1e2d3d;
    --border-bright: #2a3f55;
    --teal:        #00d4aa;
    --teal-dim:    #00a882;
    --teal-glow:   rgba(0,212,170,0.15);
    --teal-glow2:  rgba(0,212,170,0.06);
    --amber:       #f59e0b;
    --red:         #ef4444;
    --green:       #22c55e;
    --blue:        #3b82f6;
    --text-primary:   #e2eaf4;
    --text-secondary: #7a94ae;
    --text-dim:       #3d5a73;
    --font-mono:   'Share Tech Mono', monospace;
    --font-body:   'Barlow', sans-serif;
    --font-display:'Barlow Condensed', sans-serif;
  }

  html, body, #root {
    height: 100%;
    background: var(--bg-void);
    color: var(--text-primary);
    font-family: var(--font-body);
    font-size: 14px;
    line-height: 1.6;
    overflow-x: hidden;
  }

  /* scrollbar */
  ::-webkit-scrollbar { width: 4px; }
  ::-webkit-scrollbar-track { background: var(--bg-deep); }
  ::-webkit-scrollbar-thumb { background: var(--border-bright); border-radius: 2px; }

  .app {
    display: grid;
    grid-template-rows: 56px 1fr;
    grid-template-columns: 240px 1fr;
    height: 100vh;
  }

  /* ── Header ── */
  .header {
    grid-column: 1 / -1;
    display: flex;
    align-items: center;
    padding: 0 24px;
    background: var(--bg-deep);
    border-bottom: 1px solid var(--border);
    gap: 16px;
    position: relative;
    z-index: 10;
  }
  .header-logo {
    font-family: var(--font-display);
    font-size: 18px;
    font-weight: 700;
    letter-spacing: 0.08em;
    color: var(--teal);
    text-transform: uppercase;
  }
  .header-sub {
    font-family: var(--font-mono);
    font-size: 11px;
    color: var(--text-dim);
    letter-spacing: 0.05em;
  }
  .header-dot {
    width: 6px; height: 6px;
    border-radius: 50%;
    background: var(--teal);
    box-shadow: 0 0 8px var(--teal);
    animation: pulse 2s ease-in-out infinite;
    margin-left: auto;
  }
  .header-status {
    font-family: var(--font-mono);
    font-size: 11px;
    color: var(--teal);
  }
  @keyframes pulse {
    0%, 100% { opacity: 1; box-shadow: 0 0 8px var(--teal); }
    50%       { opacity: 0.4; box-shadow: 0 0 3px var(--teal); }
  }

  /* ── Sidebar ── */
  .sidebar {
    background: var(--bg-deep);
    border-right: 1px solid var(--border);
    padding: 24px 0;
    display: flex;
    flex-direction: column;
    gap: 4px;
    overflow-y: auto;
  }
  .sidebar-label {
    font-family: var(--font-mono);
    font-size: 10px;
    color: var(--text-dim);
    letter-spacing: 0.12em;
    text-transform: uppercase;
    padding: 0 20px 8px;
    margin-top: 16px;
  }
  .sidebar-item {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 9px 20px;
    cursor: pointer;
    color: var(--text-secondary);
    font-weight: 500;
    font-size: 13px;
    border-left: 2px solid transparent;
    transition: all 0.15s;
  }
  .sidebar-item:hover { background: var(--bg-hover); color: var(--text-primary); }
  .sidebar-item.active {
    color: var(--teal);
    border-left-color: var(--teal);
    background: var(--teal-glow2);
  }
  .sidebar-icon { font-size: 15px; width: 18px; text-align: center; }

  /* ── Main ── */
  .main {
    background: var(--bg-void);
    overflow-y: auto;
    padding: 28px 32px;
    display: flex;
    flex-direction: column;
    gap: 24px;
  }

  /* ── Page title ── */
  .page-title {
    font-family: var(--font-display);
    font-size: 22px;
    font-weight: 700;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    color: var(--text-primary);
  }
  .page-title span { color: var(--teal); }
  .page-desc {
    font-size: 13px;
    color: var(--text-secondary);
    margin-top: 4px;
  }

  /* ── Cards ── */
  .card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 20px 24px;
  }
  .card-title {
    font-family: var(--font-display);
    font-size: 13px;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--teal);
    margin-bottom: 16px;
    display: flex;
    align-items: center;
    gap: 8px;
  }
  .card-title::before {
    content: '';
    display: block;
    width: 3px; height: 14px;
    background: var(--teal);
    border-radius: 2px;
  }

  /* ── Grid layouts ── */
  .grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
  .grid-3 { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 16px; }
  .grid-4 { display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; }

  /* ── Stat cards ── */
  .stat-card {
    background: var(--bg-panel);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 16px 20px;
  }
  .stat-label {
    font-family: var(--font-mono);
    font-size: 10px;
    color: var(--text-dim);
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 8px;
  }
  .stat-value {
    font-family: var(--font-display);
    font-size: 28px;
    font-weight: 700;
    color: var(--teal);
    line-height: 1;
  }
  .stat-sub {
    font-size: 11px;
    color: var(--text-secondary);
    margin-top: 4px;
  }

  /* ── Form ── */
  .form-group { margin-bottom: 14px; }
  .form-label {
    display: block;
    font-family: var(--font-mono);
    font-size: 11px;
    color: var(--text-secondary);
    letter-spacing: 0.08em;
    margin-bottom: 6px;
    text-transform: uppercase;
  }
  .form-input, .form-select, .form-textarea {
    width: 100%;
    background: var(--bg-panel);
    border: 1px solid var(--border);
    border-radius: 4px;
    color: var(--text-primary);
    font-family: var(--font-body);
    font-size: 13px;
    padding: 8px 12px;
    outline: none;
    transition: border-color 0.15s;
  }
  .form-input:focus, .form-select:focus, .form-textarea:focus {
    border-color: var(--teal-dim);
    box-shadow: 0 0 0 2px var(--teal-glow);
  }
  .form-textarea {
    font-family: var(--font-mono);
    font-size: 12px;
    resize: vertical;
    min-height: 280px;
  }

  /* ── Buttons ── */
  .btn {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 9px 20px;
    border-radius: 4px;
    font-family: var(--font-display);
    font-size: 13px;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    cursor: pointer;
    border: none;
    transition: all 0.15s;
  }
  .btn-primary {
    background: var(--teal);
    color: var(--bg-void);
  }
  .btn-primary:hover { background: #00f0c0; box-shadow: 0 0 16px var(--teal-glow); }
  .btn-primary:disabled { opacity: 0.4; cursor: not-allowed; }
  .btn-ghost {
    background: transparent;
    color: var(--text-secondary);
    border: 1px solid var(--border);
  }
  .btn-ghost:hover { border-color: var(--border-bright); color: var(--text-primary); }
  .btn-danger {
    background: transparent;
    color: var(--red);
    border: 1px solid transparent;
  }
  .btn-danger:hover { border-color: var(--red); background: rgba(239,68,68,0.08); }

  /* ── Status badges ── */
  .badge {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    padding: 3px 10px;
    border-radius: 3px;
    font-family: var(--font-mono);
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
  }
  .badge-pending  { background: rgba(245,158,11,0.12); color: var(--amber); border: 1px solid rgba(245,158,11,0.3); }
  .badge-running  { background: rgba(59,130,246,0.12); color: var(--blue);  border: 1px solid rgba(59,130,246,0.3); }
  .badge-complete { background: rgba(34,197,94,0.12);  color: var(--green); border: 1px solid rgba(34,197,94,0.3); }
  .badge-failed   { background: rgba(239,68,68,0.12);  color: var(--red);   border: 1px solid rgba(239,68,68,0.3); }
  .badge-dot { width: 5px; height: 5px; border-radius: 50%; background: currentColor; }
  .badge-running .badge-dot { animation: pulse 1.2s ease-in-out infinite; }

  /* ── Table ── */
  .table-wrap { overflow-x: auto; }
  table { width: 100%; border-collapse: collapse; }
  th {
    font-family: var(--font-mono);
    font-size: 10px;
    color: var(--text-dim);
    letter-spacing: 0.1em;
    text-transform: uppercase;
    text-align: left;
    padding: 8px 14px;
    border-bottom: 1px solid var(--border);
  }
  td {
    padding: 10px 14px;
    border-bottom: 1px solid var(--border);
    font-size: 13px;
    color: var(--text-secondary);
    vertical-align: middle;
  }
  tr:last-child td { border-bottom: none; }
  tr:hover td { background: var(--bg-hover); color: var(--text-primary); }
  .mono { font-family: var(--font-mono); font-size: 11px; }

  /* ── Job tracker ── */
  .job-card {
    background: var(--bg-panel);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 16px 20px;
    display: flex;
    align-items: center;
    gap: 16px;
    margin-bottom: 10px;
  }
  .job-card.complete { border-color: rgba(34,197,94,0.25); }
  .job-card.failed   { border-color: rgba(239,68,68,0.25); }
  .job-card.running  { border-color: rgba(59,130,246,0.25); }
  .job-id { font-family: var(--font-mono); font-size: 11px; color: var(--text-dim); flex: 1; }
  .job-method { font-weight: 600; color: var(--text-primary); }
  .job-features { font-size: 12px; color: var(--text-secondary); }

  /* ── Result metrics ── */
  .metric-row {
    display: flex;
    gap: 12px;
    flex-wrap: wrap;
    margin-top: 12px;
  }
  .metric-chip {
    background: var(--bg-panel);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 6px 14px;
    font-family: var(--font-mono);
    font-size: 12px;
    color: var(--text-secondary);
  }
  .metric-chip span { color: var(--teal); font-weight: 600; }

  /* ── Empty state ── */
  .empty {
    text-align: center;
    padding: 48px 24px;
    color: var(--text-dim);
    font-family: var(--font-mono);
    font-size: 12px;
    letter-spacing: 0.05em;
  }
  .empty-icon { font-size: 32px; margin-bottom: 12px; opacity: 0.4; }

  /* ── Spinner ── */
  .spinner {
    width: 14px; height: 14px;
    border: 2px solid var(--border);
    border-top-color: var(--teal);
    border-radius: 50%;
    animation: spin 0.7s linear infinite;
    display: inline-block;
  }
  @keyframes spin { to { transform: rotate(360deg); } }

  /* ── Alert ── */
  .alert {
    padding: 12px 16px;
    border-radius: 4px;
    font-size: 13px;
    display: flex;
    align-items: center;
    gap: 10px;
  }
  .alert-error   { background: rgba(239,68,68,0.1);  border: 1px solid rgba(239,68,68,0.3);  color: #fca5a5; }
  .alert-success { background: rgba(34,197,94,0.1);  border: 1px solid rgba(34,197,94,0.3);  color: #86efac; }
  .alert-info    { background: rgba(59,130,246,0.1); border: 1px solid rgba(59,130,246,0.3); color: #93c5fd; }

  /* ── Divider ── */
  .divider { border: none; border-top: 1px solid var(--border); margin: 8px 0; }

  /* ── Teal highlight text ── */
  .teal { color: var(--teal); }
  .dim  { color: var(--text-dim); }
`;

// ─── Default JSON config ───────────────────────────────────────────────────────
const DEFAULT_JSON = JSON.stringify({
    data: {
        params: [
            { name: "min_pred_norm", value: -1.3875, valueType: 2 },
            { name: "max_pred_norm", value: 1.3875, valueType: 2 },
            { name: "min_target_norm", value: 0.005, valueType: 2 },
            { name: "max_target_norm", value: 0.995, valueType: 2 },
            { name: "plot", value: 1, valueType: 1 },
            { name: "fi_random_state", value: 5195, valueType: 2 },
            { name: "fi_n_estimators", value: 100, valueType: 2 },
            { name: "fi_max_depth", value: 25, valueType: 2 },
            { name: "fi_max_features", value: 3, valueType: 2 },
            { name: "n_range_start", value: 2, valueType: 2 },
            { name: "n_range_end", value: 30, valueType: 2 },
            { name: "kmeans_random_state_range_start", value: 0, valueType: 2 },
            { name: "kmeans_random_state_range_end", value: 50, valueType: 2 },
            { name: "gmm_random_state_range_start", value: 0, valueType: 2 },
            { name: "gmm_random_state_range_end", value: 50, valueType: 2 },
            { name: "target_feature", value: "Production", valueType: 0 },
            { name: "split_seed_range_start", value: 0, valueType: 2 },
            { name: "split_seed_range_end", value: 10, valueType: 2 },
            { name: "rf_seed_range_start", value: 0, valueType: 2 },
            { name: "rf_seed_range_end", value: 10, valueType: 2 },
            { name: "run_sampling_split", value: 1, valueType: 1 },
            { name: "run_test", value: 1, valueType: 1 },
            { name: "min_plot", value: -1.5, valueType: 2 },
            { name: "max_plot", value: 1.5, valueType: 2 },
            { name: "auto_select_features", value: 0, valueType: 1 },
            { name: "mi_threshold", value: 0.1, valueType: 2 },
            { name: "variance_threshold", value: 0.95, valueType: 2 },
            { name: "run_bnn", value: 0, valueType: 1 },
            { name: "bnn_library", value: "pytorch", valueType: 0 },
            { name: "bnn_n_samples", value: 400, valueType: 2 },
            { name: "bnn_burn_in", value: 0.85, valueType: 2 },
        ],
        tables: [
            { name: "predictive_features", values: ["Por", "Brittle", "Latitude", "Longitude"] },
            { name: "bnn_hidden_neurons", values: [9] },
        ],
    },
    metadata: {
        calculatorName: "aiProductionPrediction",
        jobName: "AI Production Prediction Workflow",
    },
}, null, 2);

// ─── API helpers ──────────────────────────────────────────────────────────────
async function apiPost(path, body) {
    const r = await fetch(API_BASE + path, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
    });
    if (!r.ok) throw new Error(`API error ${r.status}`);
    return r.json();
}
async function apiGet(path) {
    const r = await fetch(API_BASE + path);
    if (!r.ok) throw new Error(`API error ${r.status}`);
    return r.json();
}
async function apiDelete(path) {
    const r = await fetch(API_BASE + path, { method: "DELETE" });
    if (!r.ok) throw new Error(`API error ${r.status}`);
    return r.json();
}

// ─── Status badge ─────────────────────────────────────────────────────────────
function StatusBadge({ status }) {
    return (
        <span className={`badge badge-${status}`}>
            <span className="badge-dot" />
            {status}
        </span>
    );
}

// ─── Submit page ──────────────────────────────────────────────────────────────
function SubmitPage({ onJobSubmitted }) {
    const [jsonText, setJsonText] = useState(DEFAULT_JSON);
    const [localTesing, setLocalTesing] = useState(true);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    const [submitted, setSubmitted] = useState(null);

    async function handleSubmit() {
        setError(null);
        setSubmitted(null);
        setLoading(true);
        try {
            const inJson = JSON.parse(jsonText);
            const result = await apiPost("/calculate", { inJson, localTesing });
            setSubmitted(result);
            onJobSubmitted(result);
        } catch (e) {
            setError(e.message);
        } finally {
            setLoading(false);
        }
    }

    return (
        <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
            <div>
                <div className="page-title">Submit <span>Pipeline Run</span></div>
                <div className="page-desc">Configure parameters and kick off an async production prediction job</div>
            </div>

            <div className="grid-2" style={{ alignItems: "start" }}>
                <div className="card">
                    <div className="card-title">Pipeline Configuration</div>
                    <div className="form-group">
                        <label className="form-label">Input JSON</label>
                        <textarea
                            className="form-textarea"
                            value={jsonText}
                            onChange={e => setJsonText(e.target.value)}
                            spellCheck={false}
                        />
                    </div>
                    <div className="form-group" style={{ display: "flex", alignItems: "center", gap: 10 }}>
                        <input
                            type="checkbox"
                            id="localTesing"
                            checked={localTesing}
                            onChange={e => setLocalTesing(e.target.checked)}
                            style={{ accentColor: "var(--teal)" }}
                        />
                        <label htmlFor="localTesing" className="form-label" style={{ margin: 0 }}>
                            Local testing mode (SQLite)
                        </label>
                    </div>
                    <button className="btn btn-primary" onClick={handleSubmit} disabled={loading}>
                        {loading ? <span className="spinner" /> : "⚡"}
                        {loading ? "Submitting..." : "Submit Job"}
                    </button>
                </div>

                <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
                    <div className="card">
                        <div className="card-title">Pipeline Overview</div>
                        {[
                            ["Feature Engineering", "Sequential normalization pipeline"],
                            ["Feature Selection", "Rank ensemble or PCA-based auto selection"],
                            ["Spatial Clustering", "GMM / K-Means silhouette sweep"],
                            ["Hyperparameter Tuning", "Parallelized RF seed sweep"],
                            ["SHAP Explainability", "TreeExplainer + GradientExplainer"],
                            ["BNN Inference", "PyTorch or TensorFlow, gated by run_bnn"],
                        ].map(([step, desc]) => (
                            <div key={step} style={{ display: "flex", gap: 10, marginBottom: 12 }}>
                                <span style={{ color: "var(--teal)", fontFamily: "var(--font-mono)", fontSize: 11, minWidth: 6 }}>›</span>
                                <div>
                                    <div style={{ fontWeight: 600, fontSize: 13 }}>{step}</div>
                                    <div style={{ fontSize: 12, color: "var(--text-secondary)" }}>{desc}</div>
                                </div>
                            </div>
                        ))}
                    </div>

                    {error && <div className="alert alert-error">⚠ {error}</div>}
                    {submitted && (
                        <div className="alert alert-success">
                            ✓ Job submitted — ID: <span className="mono teal">{submitted.job_id.slice(0, 8)}…</span>
                            <br />
                            <span style={{ fontSize: 12, marginTop: 4, display: "block" }}>
                                Poll status in the <strong>Jobs</strong> tab
                            </span>
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
}

// ─── Jobs page ────────────────────────────────────────────────────────────────
function JobsPage({ jobs, onRefresh }) {
    const [polling, setPolling] = useState({});

    async function pollStatus(jobId) {
        setPolling(p => ({ ...p, [jobId]: true }));
        try {
            const result = await apiGet(`/runs/${jobId}/status`);
            onRefresh(result);
        } catch (e) {
            console.error(e);
        } finally {
            setPolling(p => ({ ...p, [jobId]: false }));
        }
    }

    if (!jobs.length) return (
        <div>
            <div className="page-title">Active <span>Jobs</span></div>
            <div className="page-desc">Monitor submitted pipeline runs in real time</div>
            <div className="card" style={{ marginTop: 20 }}>
                <div className="empty">
                    <div className="empty-icon">◈</div>
                    No jobs submitted yet. Submit a run from the Pipeline tab.
                </div>
            </div>
        </div>
    );

    return (
        <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
            <div>
                <div className="page-title">Active <span>Jobs</span></div>
                <div className="page-desc">Monitor submitted pipeline runs in real time</div>
            </div>
            <div className="card">
                <div className="card-title">Job Queue ({jobs.length})</div>
                {jobs.map(job => (
                    <div key={job.job_id} className={`job-card ${job.status}`}>
                        <StatusBadge status={job.status} />
                        <div style={{ flex: 1 }}>
                            <div className="job-id">{job.job_id}</div>
                            <div style={{ fontSize: 11, color: "var(--text-dim)", fontFamily: "var(--font-mono)", marginTop: 2 }}>
                                Submitted: {job.submitted}
                                {job.completed && ` — Completed: ${job.completed}`}
                            </div>
                            {job.mlflow_run_id && (
                                <div style={{ fontSize: 11, color: "var(--teal)", fontFamily: "var(--font-mono)", marginTop: 2 }}>
                                    MLflow: {job.mlflow_run_id.slice(0, 16)}…
                                </div>
                            )}
                            {job.error && (
                                <div style={{ fontSize: 12, color: "var(--red)", marginTop: 4 }}>⚠ {job.error}</div>
                            )}
                        </div>
                        <button
                            className="btn btn-ghost"
                            onClick={() => pollStatus(job.job_id)}
                            disabled={polling[job.job_id]}
                            style={{ fontSize: 12, padding: "6px 14px" }}
                        >
                            {polling[job.job_id] ? <span className="spinner" /> : "↻"} Refresh
                        </button>
                    </div>
                ))}
            </div>
        </div>
    );
}

// ─── Results page ─────────────────────────────────────────────────────────────
function ResultsPage() {
    const [runId, setRunId] = useState("");
    const [localTesing, setLocalTesing] = useState(true);
    const [result, setResult] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

    async function fetchResult() {
        if (!runId.trim()) return;
        setError(null); setResult(null); setLoading(true);
        try {
            const data = await apiGet(`/results/${runId.trim()}?localTesing=${localTesing}`);
            setResult(data);
        } catch (e) {
            setError(e.message);
        } finally {
            setLoading(false);
        }
    }

    return (
        <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
            <div>
                <div className="page-title">Run <span>Results</span></div>
                <div className="page-desc">Retrieve pipeline results from the database by MLflow run ID</div>
            </div>
            <div className="card">
                <div className="card-title">Query Results</div>
                <div style={{ display: "flex", gap: 10, alignItems: "flex-end", marginBottom: 16 }}>
                    <div style={{ flex: 1 }}>
                        <label className="form-label">MLflow Run ID</label>
                        <input
                            className="form-input"
                            placeholder="e.g. 9077ff5253764624b01ffb4fb1ab19f7"
                            value={runId}
                            onChange={e => setRunId(e.target.value)}
                            onKeyDown={e => e.key === "Enter" && fetchResult()}
                        />
                    </div>
                    <button className="btn btn-primary" onClick={fetchResult} disabled={loading}>
                        {loading ? <span className="spinner" /> : "⊕"} Fetch
                    </button>
                </div>
                <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                    <input type="checkbox" id="lt2" checked={localTesing} onChange={e => setLocalTesing(e.target.checked)} style={{ accentColor: "var(--teal)" }} />
                    <label htmlFor="lt2" className="form-label" style={{ margin: 0 }}>Local testing mode</label>
                </div>
            </div>

            {error && <div className="alert alert-error">⚠ {error}</div>}

            {result && (
                <div className="card">
                    <div className="card-title">Run Output</div>
                    <div className="grid-2" style={{ marginBottom: 16 }}>
                        <div>
                            <div className="form-label">Best Sampling Method</div>
                            <div style={{ fontFamily: "var(--font-display)", fontSize: 18, fontWeight: 700, color: "var(--teal)" }}>
                                {result.best_sampling_method || "—"}
                            </div>
                        </div>
                        <div>
                            <div className="form-label">Status</div>
                            <StatusBadge status={result.status || "complete"} />
                        </div>
                    </div>
                    <hr className="divider" />
                    <div className="metric-row">
                        {[
                            ["Max Depth", result.max_depth],
                            ["Num Trees", result.num_trees],
                            ["Max Features", result.max_features],
                            ["Split Seed", result.split_seed],
                            ["RF Seed", result.rf_seed],
                        ].map(([label, val]) => (
                            <div className="metric-chip" key={label}>
                                {label}: <span>{val ?? "—"}</span>
                            </div>
                        ))}
                    </div>
                    <div style={{ marginTop: 16 }}>
                        <div className="form-label">Selected Features</div>
                        <div style={{ fontFamily: "var(--font-mono)", fontSize: 12, color: "var(--teal)", marginTop: 4 }}>
                            {result.selected_features || "—"}
                        </div>
                    </div>
                    <div style={{ marginTop: 12 }}>
                        <div className="form-label">Timestamp</div>
                        <div className="mono dim">{result.timestamp || "—"}</div>
                    </div>
                    <div style={{ marginTop: 12 }}>
                        <div className="form-label">MLflow Run ID</div>
                        <div className="mono dim">{result.mlflow_run_id || "—"}</div>
                    </div>
                </div>
            )}
        </div>
    );
}

// ─── History page ─────────────────────────────────────────────────────────────
function HistoryPage() {
    const [runs, setRuns] = useState([]);
    const [loading, setLoading] = useState(false);
    const [localTesing, setLocalTesing] = useState(true);
    const [error, setError] = useState(null);
    const [deleting, setDeleting] = useState(null);

    async function fetchRuns() {
        setLoading(true); setError(null);
        try {
            const data = await apiGet(`/runs?localTesing=${localTesing}&limit=50`);
            setRuns(data);
        } catch (e) {
            setError(e.message);
        } finally {
            setLoading(false);
        }
    }

    async function deleteRun(runId) {
        setDeleting(runId);
        try {
            await apiDelete(`/runs/${runId}?localTesing=${localTesing}`);
            setRuns(r => r.filter(x => x.mlflow_run_id !== runId));
        } catch (e) {
            setError(e.message);
        } finally {
            setDeleting(null);
        }
    }

    useEffect(() => { fetchRuns(); }, [localTesing]);

    return (
        <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
            <div style={{ display: "flex", alignItems: "flex-end", justifyContent: "space-between" }}>
                <div>
                    <div className="page-title">Run <span>History</span></div>
                    <div className="page-desc">All past pipeline runs persisted to the SQL database</div>
                </div>
                <div style={{ display: "flex", gap: 10, alignItems: "center" }}>
                    <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                        <input type="checkbox" id="lt3" checked={localTesing} onChange={e => setLocalTesing(e.target.checked)} style={{ accentColor: "var(--teal)" }} />
                        <label htmlFor="lt3" className="form-label" style={{ margin: 0 }}>Local mode</label>
                    </div>
                    <button className="btn btn-ghost" onClick={fetchRuns} disabled={loading}>
                        {loading ? <span className="spinner" /> : "↻"} Refresh
                    </button>
                </div>
            </div>

            {error && <div className="alert alert-error">⚠ {error}</div>}

            <div className="card" style={{ padding: 0, overflow: "hidden" }}>
                <div className="table-wrap">
                    {!runs.length ? (
                        <div className="empty">
                            <div className="empty-icon">◈</div>
                            No runs found in database.
                        </div>
                    ) : (
                        <table>
                            <thead>
                                <tr>
                                    <th>Timestamp</th>
                                    <th>Sampling Method</th>
                                    <th>Selection Mode</th>
                                    <th>Max Depth</th>
                                    <th>Num Trees</th>
                                    <th>MLflow Run ID</th>
                                    <th></th>
                                </tr>
                            </thead>
                            <tbody>
                                {runs.map(run => (
                                    <tr key={run.mlflow_run_id}>
                                        <td className="mono">{run.timestamp}</td>
                                        <td style={{ color: "var(--teal)", fontWeight: 600 }}>{run.best_sampling_method}</td>
                                        <td>{run.selection_mode}</td>
                                        <td className="mono">{run.max_depth}</td>
                                        <td className="mono">{run.num_trees}</td>
                                        <td className="mono" style={{ color: "var(--text-dim)", fontSize: 10 }}>
                                            {run.mlflow_run_id?.slice(0, 12)}…
                                        </td>
                                        <td>
                                            <button
                                                className="btn btn-danger"
                                                style={{ padding: "4px 10px", fontSize: 11 }}
                                                onClick={() => deleteRun(run.mlflow_run_id)}
                                                disabled={deleting === run.mlflow_run_id}
                                            >
                                                {deleting === run.mlflow_run_id ? <span className="spinner" /> : "✕"}
                                            </button>
                                        </td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    )}
                </div>
            </div>
        </div>
    );
}

// ─── Health page ──────────────────────────────────────────────────────────────
function HealthPage() {
    const [health, setHealth] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

    async function check() {
        setLoading(true); setError(null);
        try {
            const data = await apiGet("/health");
            setHealth(data);
        } catch (e) {
            setError("API unreachable — is the FastAPI server running?");
        } finally {
            setLoading(false);
        }
    }

    useEffect(() => { check(); }, []);

    return (
        <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
            <div>
                <div className="page-title">System <span>Health</span></div>
                <div className="page-desc">API and service connectivity status</div>
            </div>
            <div className="grid-2">
                <div className="card">
                    <div className="card-title">API Status</div>
                    {loading && <div className="alert alert-info"><span className="spinner" /> Checking…</div>}
                    {error && <div className="alert alert-error">⚠ {error}</div>}
                    {health && (
                        <div>
                            <div className="alert alert-success" style={{ marginBottom: 12 }}>✓ API online</div>
                            <div className="metric-chip" style={{ display: "inline-block" }}>
                                Last checked: <span>{health.timestamp}</span>
                            </div>
                        </div>
                    )}
                    <button className="btn btn-ghost" onClick={check} disabled={loading} style={{ marginTop: 16 }}>
                        ↻ Re-check
                    </button>
                </div>
                <div className="card">
                    <div className="card-title">Services</div>
                    {[
                        ["FastAPI", "Port 8000", health ? "online" : "unknown"],
                        ["MLflow", "Port 5000", "running"],
                        ["PostgreSQL", "Port 5432", "running"],
                        ["Calculator", "Container", "running"],
                    ].map(([name, port, status]) => (
                        <div key={name} style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 10 }}>
                            <div>
                                <div style={{ fontWeight: 600 }}>{name}</div>
                                <div className="mono dim">{port}</div>
                            </div>
                            <StatusBadge status={status === "online" || status === "running" ? "complete" : "pending"} />
                        </div>
                    ))}
                </div>
            </div>
        </div>
    );
}

// ─── App shell ────────────────────────────────────────────────────────────────
const NAV = [
    { id: "submit", label: "Pipeline", icon: "⚡" },
    { id: "jobs", label: "Jobs", icon: "◎" },
    { id: "results", label: "Results", icon: "◈" },
    { id: "history", label: "History", icon: "▤" },
    { id: "health", label: "Health", icon: "◉" },
];

export default function App() {
    const [page, setPage] = useState("submit");
    const [jobs, setJobs] = useState([]);

    function handleJobSubmitted(job) {
        setJobs(prev => [job, ...prev.filter(j => j.job_id !== job.job_id)]);
    }
    function handleJobRefresh(job) {
        setJobs(prev => prev.map(j => j.job_id === job.job_id ? job : j));
    }

    return (
        <>
            <style>{css}</style>
            <div className="app">
                <header className="header">
                    <div>
                        <div className="header-logo">Production Prediction ML</div>
                        <div className="header-sub">Bayesian · Random Forest · SHAP · MLflow · PostgreSQL</div>
                    </div>
                    <div className="header-dot" />
                    <div className="header-status">API LIVE</div>
                </header>

                <nav className="sidebar">
                    <div className="sidebar-label">Navigation</div>
                    {NAV.map(item => (
                        <div
                            key={item.id}
                            className={`sidebar-item ${page === item.id ? "active" : ""}`}
                            onClick={() => setPage(item.id)}
                        >
                            <span className="sidebar-icon">{item.icon}</span>
                            {item.label}
                            {item.id === "jobs" && jobs.length > 0 && (
                                <span style={{
                                    marginLeft: "auto",
                                    background: "var(--teal)",
                                    color: "var(--bg-void)",
                                    borderRadius: "10px",
                                    fontSize: "10px",
                                    padding: "1px 7px",
                                    fontFamily: "var(--font-mono)",
                                    fontWeight: 700,
                                }}>{jobs.length}</span>
                            )}
                        </div>
                    ))}

                    <div className="sidebar-label" style={{ marginTop: "auto" }}>Links</div>
                    <a
                        href="http://localhost:5000"
                        target="_blank"
                        rel="noreferrer"
                        className="sidebar-item"
                        style={{ textDecoration: "none" }}
                    >
                        <span className="sidebar-icon">◎</span> MLflow UI
                    </a>
                    <a
                        href="http://localhost:8000/docs"
                        target="_blank"
                        rel="noreferrer"
                        className="sidebar-item"
                        style={{ textDecoration: "none" }}
                    >
                        <span className="sidebar-icon">⊕</span> API Docs
                    </a>
                </nav>

                <main className="main">
                    {page === "submit" && <SubmitPage onJobSubmitted={handleJobSubmitted} />}
                    {page === "jobs" && <JobsPage jobs={jobs} onRefresh={handleJobRefresh} />}
                    {page === "results" && <ResultsPage />}
                    {page === "history" && <HistoryPage />}
                    {page === "health" && <HealthPage />}
                </main>
            </div>
        </>
    );
}