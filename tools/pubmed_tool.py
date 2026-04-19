from Bio import Entrez

Entrez.email = "boyinianjaneyulu54@gmail.com"
Entrez.tool = "medical-ai-agent"


def search_pubmed(query):
    try:
        # 🔹 Clean query (VERY IMPORTANT)
        clean_query = query.replace("latest", "").replace("research", "").strip()

        # 🔹 Step 1: Search
        handle = Entrez.esearch(
            db="pubmed",
            term=clean_query,
            retmax=3
        )

        record = Entrez.read(handle)
        ids = record.get("IdList", [])

        # ❌ If no results
        if not ids:
            return "No PubMed results found."

        # 🔹 Step 2: Fetch abstracts
        handle = Entrez.efetch(
            db="pubmed",
            id=",".join(ids),
            rettype="abstract",
            retmode="text"
        )

        abstracts = handle.read()

        return abstracts[:2000]

    except Exception as e:
        return f"PubMed error: {str(e)}"