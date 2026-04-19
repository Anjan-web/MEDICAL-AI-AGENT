import streamlit as st
import requests

# ✅ Your ngrok URL — update this when ngrok restarts
API_URL = "https://anjandata-medical-ai-agent.hf.space/ask"
st.set_page_config(
    page_title="Medical AI Agent",
    page_icon="🏥",
    layout="centered"
)

st.title("🏥 Medical AI Agent")
st.caption("Powered by RAG + PubMed + WHO Guidelines")

# Example questions
with st.expander("💡 Example questions"):
    st.markdown("""
    - What is malaria and how does it spread?
    - What are recent research findings on diabetes treatment?
    - What are WHO guidelines on tuberculosis treatment?
    - What are the side effects of quinine?
    - What are the latest guidelines on HPV vaccine?
    """)

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Clear chat button
if st.button("🗑️ Clear chat"):
    st.session_state.messages = []
    st.rerun()

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg.get("sources"):
            st.caption(f"📚 Sources: {', '.join(msg['sources'])}")
        if msg.get("confidence"):
            st.caption(f"🎯 Confidence: {msg['confidence']}")

# Chat input
if question := st.chat_input("Ask a medical question..."):

    # Show user message
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    # Call API
    with st.chat_message("assistant"):
        with st.spinner("Searching medical knowledge..."):
            try:
                response = requests.get(
                    API_URL,
                    params={"question": question},
                    timeout=60,
                    headers={
                        "ngrok-skip-browser-warning": "true",
                        "User-Agent": "medical-ai-streamlit",
                        "Accept": "application/json"
                    }
                )

                if response.status_code != 200:
                    st.error(f"API error: status {response.status_code}")
                    st.code(response.text[:300])
                else:
                    data = response.json()
                    answer = data.get("answer", "No answer returned")
                    sources = data.get("sources", [])
                    confidence = data.get("confidence", "")

                    st.write(answer)

                    col1, col2 = st.columns(2)
                    with col1:
                        if sources:
                            st.caption(f"📚 Sources: {', '.join(sources)}")
                    with col2:
                        if confidence:
                            color = "🟢" if confidence == "High" else "🟡"
                            st.caption(f"{color} Confidence: {confidence}")

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                        "confidence": confidence
                    })

            except requests.exceptions.JSONDecodeError:
                st.error("Could not parse API response — check if FastAPI server is running")
                st.code(response.text[:300])
            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to API — make sure FastAPI and ngrok are both running")
            except requests.exceptions.Timeout:
                st.error("Request timed out — the question may be complex, try again")
            except Exception as e:
                st.error(f"Unexpected error: {e}")

# Sidebar
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    **Medical AI Agent** answers medical questions using:
    
    - 📖 Medical Encyclopedia (RAG)
    - 🔬 Live PubMed Research
    - 🏥 WHO/CDC Guidelines
    """)

    st.divider()

    st.header("📊 System Info")
    st.metric("Faithfulness", "0.87")
    st.metric("Answer Relevancy", "0.96")
    st.metric("Context Recall", "1.00")

    st.divider()

    st.header("⚙️ API Status")
    if st.button("🔍 Check API"):
        try:
            r = requests.get(
                API_URL.replace("/ask", "/"),
                timeout=5,
                headers={"ngrok-skip-browser-warning": "true"}
            )
            st.success("API is online ✅")
        except Exception:
            st.error("API is offline ❌")

    st.divider()
    st.caption("Built with LangGraph + FastAPI + Streamlit")