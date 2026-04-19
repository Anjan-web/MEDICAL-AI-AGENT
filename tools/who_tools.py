import requests
from urllib.parse import quote


def fetch_whp_data(query):
    """
    WHO website blocks automated requests (403).
    Instead, search PubMed filtered to WHO/guideline publications —
    PubMed indexes WHO guidelines and returns reliable results.
    """
    try:
        clean_query = (
            query.lower()
            .replace("can you explain", "")
            .replace("in detail", "")
            .replace("tell me about", "")
            .replace("what is", "")
            .replace("what are", "")
            .strip()
        )

        # ✅ Search PubMed with guideline filter
        # This finds actual WHO/CDC/official guidelines indexed in PubMed
        guideline_query = f"{clean_query} AND (guideline[pt] OR \"WHO\" OR \"World Health Organization\" OR \"CDC\" OR recommendation)"

        print(f"🏥 WHO→PubMed guideline query: {guideline_query}")

        from Bio import Entrez
        Entrez.email = "boyinianjaneyulu54@gmail.com"
        Entrez.tool = "medical-ai-agent"

        handle = Entrez.esearch(
            db="pubmed",
            term=guideline_query,
            retmax=3,
            sort="relevance"
        )
        record = Entrez.read(handle)
        ids = record.get("IdList", [])

        print(f"🏥 WHO→PubMed found {len(ids)} guideline papers")

        if not ids:
            # Fallback — search without strict guideline filter
            handle = Entrez.esearch(
                db="pubmed",
                term=f"{clean_query} guideline recommendation",
                retmax=3,
                sort="relevance"
            )
            record = Entrez.read(handle)
            ids = record.get("IdList", [])

        if not ids:
            print("⚠️ No guideline papers found")
            return ""

        handle = Entrez.efetch(
            db="pubmed",
            id=",".join(ids),
            rettype="abstract",
            retmode="text"
        )

        abstracts = handle.read()
        print(f"✅ WHO→PubMed returned {len(abstracts)} chars")
        return abstracts[:3000]

    except Exception as e:
        print(f"⚠️ WHO tool error: {e}")
        return ""