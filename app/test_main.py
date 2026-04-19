from app.main import graph

questions = [
    # RAG - General Medical
    ("RAG", "What is malaria and how does it spread?"),
    ("RAG", "What are the symptoms of appendicitis?"),
    ("RAG", "What causes type 2 diabetes?"),
    ("RAG", "What is dengue fever?"),
    ("RAG", "How does tuberculosis affect the lungs?"),
    # Drug / Treatment
    ("DRUG", "What is the treatment for falciparum malaria?"),
    ("DRUG", "What medications are used to treat tuberculosis?"),
    ("DRUG", "What does chloroquine treat and what are its side effects?"),
    ("DRUG", "How is typhoid fever treated?"),
    ("DRUG", "What are the side effects of quinine?"),
    # PubMed - Latest Research
    ("PUBMED", "What are recent research findings on diabetes treatment?"),
    ("PUBMED", "What are the latest clinical trials for tuberculosis cure?"),
    ("PUBMED", "What does recent research say about dengue vaccine development?"),
    ("PUBMED", "What are new findings on antibiotic resistance in 2025?"),
    ("PUBMED", "What are recent studies on malaria drug resistance?"),
    # WHO - Guidelines
    ("WHO", "What are WHO guidelines on tuberculosis treatment?"),
    ("WHO", "What are the latest guidelines on childhood vaccination schedule?"),
    ("WHO", "What are WHO recommendations on diabetes management?"),
    ("WHO", "What are current CDC guidelines on malaria prevention for travelers?"),
    ("WHO", "What are WHO guidelines on dengue fever management?"),
]

print("=" * 70)
print(f"{'#':<4} {'EXP':<7} {'GOT':<10} {'SOURCES':<25} {'STATUS':<6} QUESTION")
print("=" * 70)

passed = 0
failed = 0

for i, (expected_intent, question) in enumerate(questions, 1):
    try:
        result = graph.invoke({"question": question})
        answer  = result.get("answer", "")
        sources = result.get("sources", [])

        source_tags = []
        if any("Page" in s for s in sources):
            source_tags.append("RAG")
        if "PubMed" in sources:
            source_tags.append("PubMed")
        if "WHO" in sources:
            source_tags.append("WHO")

        source_str = "+".join(source_tags) if source_tags else "NONE"

        # Determine actual intent from nodes print (approximation)
        if "PubMed" in sources and "WHO" not in sources:
            got_intent = "PUBMED"
        elif "WHO" in sources:
            got_intent = "WHO"
        elif any("Page" in s for s in sources):
            got_intent = "RAG"
        else:
            got_intent = "NONE"

        if len(answer) < 100 or answer.strip() == "Not enough information":
            status = "FAIL"
            failed += 1
        else:
            status = "OK"
            passed += 1

        print(f"[{i:02d}] {expected_intent:<7} {got_intent:<10} {source_str:<25} {status:<6} {question[:45]}")

    except Exception as e:
        print(f"[{i:02d}] {expected_intent:<7} {'ERROR':<10} {'':25} {'FAIL':<6} {question[:45]}")
        print(f"      Exception: {e}")
        failed += 1

print("=" * 70)
print(f"PASSED: {passed}/20    FAILED: {failed}/20")
print("=" * 70)