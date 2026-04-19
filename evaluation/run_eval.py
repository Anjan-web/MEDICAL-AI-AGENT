import os
import csv
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI

# ✅ FIXED: use llm_factory as RAGAS instructs
from ragas.llms import llm_factory

# ✅ FIXED: import from ragas.metrics (not collections) — collections
# requires a different internal init that breaks with llm= kwarg
from ragas.metrics import Faithfulness, AnswerRelevancy, ContextRecall
from ragas import evaluate, RunConfig
from datasets import Dataset

from app.main import graph
from evaluation.test_data import test_data

# --------------------------------------------------
# STEP 0: Load environment
# --------------------------------------------------
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not set in .env file")

# --------------------------------------------------
# STEP 1: Set up evaluator LLM
# ✅ FIXED: llm_factory is what this RAGAS version wants
# --------------------------------------------------
openai_client = OpenAI(api_key=OPENAI_API_KEY)

evaluator_llm = llm_factory(
    "gpt-4o-mini",
    client=openai_client,
    max_tokens=4096
)

# --------------------------------------------------
# STEP 2: Run your AI agent on all test questions
# --------------------------------------------------
results = []
print("--- Running Graph for Test Data ---\n")

for i, item in enumerate(test_data):
    print(f"  [{i+1}/{len(test_data)}] {item['question'][:60]}...")
    try:
        response = graph.invoke({"question": item["question"]})
        results.append({
            "question":     item["question"],
            "answer":       response.get("answer", "No answer generated"),
            "contexts":     [str(response.get("context", ""))],
            "ground_truth": item["ground_truth"]
        })
    except Exception as e:
        print(f"  ⚠️  Failed: {e}")
        results.append({
            "question":     item["question"],
            "answer":       "ERROR",
            "contexts":     [""],
            "ground_truth": item["ground_truth"]
        })

dataset = Dataset.from_list(results)

# --------------------------------------------------
# STEP 3: Initialize metrics
# ✅ FIXED: pass llm via .llm attribute after instantiation
# This is what this RAGAS version actually supports
# --------------------------------------------------
faithfulness_metric = Faithfulness()
faithfulness_metric.llm = evaluator_llm

answer_relevancy_metric = AnswerRelevancy()
answer_relevancy_metric.llm = evaluator_llm

context_recall_metric = ContextRecall()
context_recall_metric.llm = evaluator_llm

# --------------------------------------------------
# STEP 4: Run evaluation
# --------------------------------------------------
print("\n--- Starting Evaluation ---")

scores = evaluate(
    dataset,
    metrics=[
        faithfulness_metric,
        answer_relevancy_metric,
        context_recall_metric
    ],
    run_config=RunConfig(timeout=120, max_retries=3)
)

# --------------------------------------------------
# STEP 5: Print formatted results + save to CSV
# --------------------------------------------------
score_dict = scores.to_pandas().mean(numeric_only=True).to_dict()

faithfulness   = round(score_dict.get("faithfulness", 0), 4)
answer_rel     = round(score_dict.get("answer_relevancy", 0), 4)
context_recall = round(score_dict.get("context_recall", 0), 4)

print("\n" + "="*45)
print("  EVALUATION RESULTS")
print("="*45)
print(f"  Faithfulness      : {faithfulness}")
print(f"  Answer Relevancy  : {answer_rel}")
print(f"  Context Recall    : {context_recall}")
print("="*45)

# Save to CSV for run history
csv_path = "evaluation/eval_history.csv"
file_exists = os.path.exists(csv_path)

with open(csv_path, "a", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=[
        "timestamp", "faithfulness", "answer_relevancy", "context_recall", "num_questions"
    ])
    if not file_exists:
        writer.writeheader()
    writer.writerow({
        "timestamp":        datetime.now().strftime("%Y-%m-%d %H:%M"),
        "faithfulness":     faithfulness,
        "answer_relevancy": answer_rel,
        "context_recall":   context_recall,
        "num_questions":    len(results)
    })

print(f"\n✅ Results saved to {csv_path}")