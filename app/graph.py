from langgraph.graph import StateGraph, END
from app.schemas import AgentState
from app.modules.retriever import get_trusted_context
from app.modules.generator import generate_rag_answer
from app.modules.evaluator import check_hallucination
from app.modules.evaluator import get_ragas_metrics
from app.modules.router import semantic_router
from app.modules.cache import get_semantic_cache, save_to_cache

async def route_node(state: AgentState):
    # This calls the Gemini-powered router we built
    category = await semantic_router(state["question"])
    return {"category": category, "logs": state["logs"] + [f"Routed to: {category}"]}

# Conditional Logic
def decide_path(state: AgentState):
    if state["category"] == "greeting":
        return "greeting_path"
    if state["category"] == "off_topic":
        return "reject"
    return "rag_path"    

# Node 1: Retrieve
async def retrieve_node(state: AgentState):
    context = get_trusted_context(state["question"])
    return {"context": "Retrieved info...", "retry_count": state["retry_count"] + 1}

# Node 2: Generate
async def generate_node(state: AgentState):
    # Check Semantic Cache first
    cached_response = await get_semantic_cache(state["question"])
    if cached_response:
        return {"answer": cached_response, "logs": state["logs"] + ["Semantic Cache Hit!"]}
    # 2. Call Generator and Unpack results
    result = await generate_rag_answer(state["question"], state["context"])
    # Ensure 'ans' matches what you pass to save_to_cache
    ans, tokens, cost = await generate_rag_answer(state["question"], state["context"])
    
    # 3. Save to Cache for future users
    save_to_cache(state["question"], ans)
    
    return {
        "answer": ans,
        "total_tokens": state.get("total_tokens", 0) + tokens,
        "total_cost": state.get("total_cost", 0.0) + cost
    }

# Node 3: Evaluate
def evaluate_node(state: AgentState):
    f_score = get_ragas_metrics(state)
    eval_result = check_hallucination(state["context"], state["answer"])
    return {
        "faithfulness_score": f_score,
        "is_hallucinated": eval_result.is_hallucinated,
        "logs": state["logs"] + [f"Evaluation: {eval_result.reasoning}"]
    }

# Logic for branching
def decide_next_step(state: AgentState):
    if state["is_hallucinated"] and state["retry_count"] < 1:
        return "retry"
    return "end"

# Build Graph
workflow = StateGraph(AgentState)
workflow.add_node("router", route_node)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("generate", generate_node)

workflow.set_entry_point("router")
workflow.add_conditional_edges("router", decide_path, {
    "greeting_path": "generate",
    "rag_path": "retrieve"
})
workflow.add_edge("retrieve", "generate")
workflow.add_node("evaluate", evaluate_node)

workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", "evaluate")

workflow.add_conditional_edges(
    "evaluate",
    decide_next_step,
    {"retry": "generate", "end": END}
)

graph_app = workflow.compile()