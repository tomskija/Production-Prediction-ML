############################################################################################
import sys
import os
import uuid
import asyncio
from datetime import datetime
from os.path import dirname, abspath, join
sys.path.insert(0, join(dirname(abspath(__file__)), 'productionPredictionCalculator'))
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from productionPredictionCalculator.Calculator import calculate
from productionPredictionCalculator.utils.dbConfig import DBConfig

############################################################################################
app = FastAPI(
    title       = "Production Prediction ML API",
    description = "REST API for the Production Prediction ML pipeline — async job submission, results retrieval, and run management.",
    version     = "1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

############################################################################################
# In-memory job status store — tracks running/pending/complete/failed jobs by job_id
# Persisted results live in SQL; this is just the status layer
job_store = {}

############################################################################################
class CalculateRequest(BaseModel):
    inJson:       dict
    localTesing:  Optional[bool] = True

############################################################################################
class JobStatusResponse(BaseModel):
    job_id:     str
    status:     str
    submitted:  str
    completed:  Optional[str] = None
    error:      Optional[str] = None
    mlflow_run_id: Optional[str] = None

############################################################################################
class RunResult(BaseModel):
    mlflow_run_id:        Optional[str]
    timestamp:            Optional[str]
    status:               Optional[str]
    selection_mode:       Optional[str]
    selected_features:    Optional[str]
    best_sampling_method: Optional[str]
    max_depth:            Optional[int]
    num_trees:            Optional[int]
    max_features:         Optional[int]
    split_seed:           Optional[int]
    rf_seed:              Optional[int]

############################################################################################
def run_pipeline(job_id='', inJson={}, localTesing=True):
    """
    Background task — runs calculate() and updates job_store on completion.
    Called via FastAPI BackgroundTasks so it runs async without blocking the API.
    """
    ########################################################################################
    try:
        job_store[job_id]['status'] = 'running'
        output_wrapper              = calculate(inJson=inJson, localTesing=localTesing)
        ########################################################################################
        if isinstance(output_wrapper, str):
            job_store[job_id]['status']    = 'failed'
            job_store[job_id]['error']     = output_wrapper
            job_store[job_id]['completed'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            return
        ########################################################################################
        mlflow_run_id = next((p['value'] for p in output_wrapper.params if p['name'] == 'mlflow_run_id'), None)
        job_store[job_id]['status']        = 'complete'
        job_store[job_id]['completed']     = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        job_store[job_id]['mlflow_run_id'] = mlflow_run_id
    except Exception as e:
        job_store[job_id]['status']    = 'failed'
        job_store[job_id]['error']     = str(e)
        job_store[job_id]['completed'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

############################################################################################
@app.get("/health")
def health():
    """
    Health check — confirms API is running.
    """
    return {"status": "ok", "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

############################################################################################
@app.post("/calculate", response_model=JobStatusResponse)
def submit_job(request: CalculateRequest, background_tasks: BackgroundTasks):
    """
    Submit a pipeline run — returns a job_id immediately.
    Pipeline runs asynchronously in the background.
    Poll /runs/{job_id}/status to check progress.
    """
    ########################################################################################
    job_id = str(uuid.uuid4())
    job_store[job_id] = {
        'job_id':        job_id,
        'status':        'pending',
        'submitted':     datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'completed':     None,
        'error':         None,
        'mlflow_run_id': None,
    }
    background_tasks.add_task(run_pipeline, job_id=job_id, inJson=request.inJson, localTesing=request.localTesing)
    return job_store[job_id]

############################################################################################
@app.get("/runs/{run_id}/status", response_model=JobStatusResponse)
def get_run_status(run_id: str):
    """
    Check status of a submitted job by job_id.
    Returns: pending / running / complete / failed
    """
    if run_id not in job_store:
        raise HTTPException(status_code=404, detail=f"Job {run_id} not found")
    return job_store[run_id]

############################################################################################
@app.get("/results/{run_id}", response_model=RunResult)
def get_results(run_id: str, localTesing: bool = True):
    """
    Get pipeline results from the database by mlflow_run_id.
    Queries run_results table directly — no MLflow dependency at read time.
    """
    ########################################################################################
    db = DBConfig(localTesing=localTesing)
    try:
        cur = db.conn.cursor()
        cur.execute("SELECT * FROM run_results WHERE mlflow_run_id = ?", (run_id,)) if localTesing else \
            cur.execute("SELECT * FROM run_results WHERE mlflow_run_id = %s", (run_id,))
        row = cur.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail=f"No results found for run_id {run_id}")
        cols   = [desc[0] for desc in cur.description]
        result = dict(zip(cols, row))
        return result
    finally:
        db.close()

############################################################################################
@app.get("/runs", response_model=list[RunResult])
def list_runs(localTesing: bool = True, limit: int = 50):
    """
    List all past pipeline runs from the database.
    Returns most recent runs first, up to limit.
    """
    ########################################################################################
    db = DBConfig(localTesing=localTesing)
    try:
        cur = db.conn.cursor()
        cur.execute("SELECT * FROM run_results ORDER BY timestamp DESC LIMIT ?", (limit,)) if localTesing else \
            cur.execute("SELECT * FROM run_results ORDER BY timestamp DESC LIMIT %s", (limit,))
        rows = cur.fetchall()
        cols = [desc[0] for desc in cur.description]
        return [dict(zip(cols, row)) for row in rows]
    finally:
        db.close()

############################################################################################
@app.delete("/runs/{run_id}")
def delete_run(run_id: str, localTesing: bool = True):
    """
    Delete a run from the database by mlflow_run_id.
    Also removes from in-memory job store if present.
    """
    ########################################################################################
    db = DBConfig(localTesing=localTesing)
    try:
        cur = db.conn.cursor()
        cur.execute("DELETE FROM run_results WHERE mlflow_run_id = ?", (run_id,)) if localTesing else \
            cur.execute("DELETE FROM run_results WHERE mlflow_run_id = %s", (run_id,))
        db.conn.commit()
        if run_id in job_store: del job_store[run_id]
        return {"deleted": run_id, "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    finally:
        db.close()

############################################################################################
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
############################################################################################