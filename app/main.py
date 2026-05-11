import warnings; warnings.filterwarnings("ignore")
import json
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from app.graph import graph_app
from pydantic import BaseModel
from fastapi import BackgroundTasks
from app.modules.evaluator import get_ragas_metrics

app = FastAPI(title="GuardianRAG API")

class QueryRequest(BaseModel):
    question: str

@app.post("/stream")
async def stream_rag(request: QueryRequest, background_tasks: BackgroundTasks):
    # This is the generator
    async def event_generator():
        initial_state = {
            "question": request.question,
            "retry_count": 0,
            "logs": [],
            "context": "",
            "answer": "",
            "is_hallucinated": False,
            "total_tokens": 0,
            "total_cost": 0.0
        }
        
        async for output in graph_app.astream(initial_state):
            yield f"data: {json.dumps(output)}\n\n"
            
            # ASYNC SHADOW EVALUATION
            if "generate" in output:
                # This runs after the user has already received the answer
                background_tasks.add_task(get_ragas_metrics, output["generate"])
        # Do NOT return a value here. Just let it finish or use a bare return. 

    # Return the response CALLING the generator
    return StreamingResponse(event_generator(), media_type="text/event-stream")
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)