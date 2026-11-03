
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from agent import evaluate_workflow

app = FastAPI(
    title="Practical 9 - AI Agent API",
    description="FastAPI backend for Agent Workflow Evaluation",
    version="1.0.0"
)

# Allow React frontend to access the backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class EvaluationRequest(BaseModel):
    goal: str = Field(
        min_length=1,
        max_length=2000,
        description="Goal to be completed by the AI agent"
    )


@app.get("/")
def home():
    return {
        "message": "Practical 9 AI Agent Backend is running!"
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "AI Agent API"
    }


@app.post("/evaluate")
def evaluate(request: EvaluationRequest):
    try:
        goal = request.goal.strip()

        if not goal:
            raise HTTPException(
                status_code=400,
                detail="Goal cannot be empty"
            )

        result = evaluate_workflow(goal)

        return {
            "success": True,
            "goal": goal,
            "evaluation": result
        }

    except HTTPException:
        raise

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=str(error)
        )
