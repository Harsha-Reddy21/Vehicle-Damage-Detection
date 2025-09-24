from fastapi import FastAPI
import uvicorn
from workflow import DamageAssessmentOrchestrator


app = FastAPI()

@app.get("/health")
def health_check():
    return {"message": "Healthy"}



@app.post("/process_claim")
def process_claim(claim: dict):
    orchestrator = DamageAssessmentOrchestrator()
    return orchestrator.process_claim(claim)



if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)



