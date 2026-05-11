import os
from dotenv import load_dotenv
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness
from langchain_google_genai import ChatGoogleGenerativeAI
from ragas.llms import LangchainLLMWrapper
from app.schemas import HallucinationCheck

load_dotenv()

# 1. Setup the Gemini model
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", 
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    max_retries=6, # Automatically retry 6 times
    timeout=60
)

# 2. Wrap Gemini for RAGAS (Prevents the OpenAI Error)
ragas_judge = LangchainLLMWrapper(llm)

# 3. The Boolean Judge (For Graph Logic)
judge_llm_structured = llm.with_structured_output(HallucinationCheck)

def check_hallucination(context: str, answer: str) -> HallucinationCheck:
    prompt = f"CONTEXT: {context}\nANSWER: {answer}\nCheck for hallucinations."
    return judge_llm_structured.invoke(prompt)

# 4. The RAGAS Scorer (For Monitoring)
def get_ragas_metrics(state):
    try:
        data = {
            "question": [state["question"]],
            "contexts": [[state["context"]]],
            "answer": [state["answer"]]
        }
        dataset = Dataset.from_dict(data)
        
        # Explicitly pass Gemini to RAGAS
        results = evaluate(
            dataset, 
            metrics=[faithfulness], 
            llm=ragas_judge
        )
        return float(results["faithfulness"])
    except Exception as e:
        print(f"Ragas Error: {e}")
        return 0.0