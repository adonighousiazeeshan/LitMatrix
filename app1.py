import streamlit as st
from extractor import extract_paper_metadata
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from pdf_processor import process_uploaded_files


# Load environment variables
load_dotenv()
@st.cache_resource
def get_llm():
    return ChatGroq(
        model_name="openai/gpt-oss-20b",
        temperature=0
    )
# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_icon="📚",
    page_title="LitMatrix - Research Assistant",
    layout="centered"
)

# --------------------------------------------------
# CUSTOM CSS STYLING FOR A MODERN UI
# --------------------------------------------------
st.markdown("""
<style>
    /* Main container padding & aesthetics */
    .block-container {
        padding-top: 2.5rem;
        padding-bottom: 3rem;
    }

    /* App Header Title */
    h1 {
        font-weight: 800 !important;
        letter-spacing: -0.5px;
    }

    /* Style buttons for a modern look */
    div.stButton > button {
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    div.stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }

    /* Expanders styling */
    .streamlit-expanderHeader {
        font-weight: 600;
        border-radius: 6px;
    }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# APP HEADER
# --------------------------------------------------
st.title("📚 LitMatrix")
st.caption("Research Intelligence Platform")

st.markdown(
    """
    <div style="font-size:16px; color: #a0a0a0; margin-bottom:15px;">
        Analyze, compare, and explore your research papers using advanced AI.
    </div>
    """,
    unsafe_allow_html=True
)

# --------------------------------------------------
# FILE UPLOADER CONTAINER (Card Layout)
# --------------------------------------------------
with st.container(border=True):
    st.markdown("### 📁 Document Upload")
    uploaded_files = st.file_uploader(
        "Upload research paper PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )

    # Process Button
    if st.button("🚀 Process Papers", use_container_width=True):
        if uploaded_files:
            with st.spinner(
                "Processing papers, generating embeddings, and storing in session memory..."
            ):
                vector_store, uploaded_filenames = process_uploaded_files(uploaded_files)

                if vector_store is None:
                    st.error("PDF processing failed. Please try again.")
                    st.stop()

                if not uploaded_filenames:
                    st.warning("No valid PDF files were processed.")
                    st.stop()
                #z

                # Save data in Streamlit session state
                st.session_state.vector_store = vector_store
                st.session_state.uploaded_filenames = uploaded_filenames
                st.session_state.messages = []
                st.session_state.extraction_results = None

            st.success("🎉 Papers indexed successfully!")

            # Show uploaded papers for confirmation inside an expander or clean list
            with st.expander("📄 View Uploaded Files List", expanded=True):
                for filename in uploaded_filenames:
                    st.markdown(f"- `{filename}`")

        else:
            st.warning(
                "⚠️ Please upload at initial PDF files before clicking Process."
            )

# --------------------------------------------------
# MAIN INTERFACE
# --------------------------------------------------
if "vector_store" in st.session_state:

    st.write("---")

    # ==================================================
    # STRUCTURED RESEARCH EXTRACTION
    # ==================================================
    with st.container(border=True):
        st.subheader("📊 Structured Research Extraction & Comparison")
        st.markdown(
            "Extract standardized fields across your uploaded papers "
            "to compare them side-by-side."
        )

        if st.button("🔍 Extract & Compare Papers", use_container_width=True):
            with st.spinner("Extracting structured details from papers..."):
                st.session_state.extraction_results = extract_paper_metadata(
                    st.session_state.vector_store,
                    st.session_state.uploaded_filenames
                )

        if (
            "extraction_results" in st.session_state
            and st.session_state.extraction_results
        ):
            st.markdown("---")
            st.dataframe(
                st.session_state.extraction_results,
                use_container_width=True
            )

    st.write("---")

    # ==================================================
    # CHAT WITH RESEARCH PAPERS
    # ==================================================
    st.subheader("💬 Chat with your Research Papers")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display previous chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # ==================================================
    # CHAT INPUT
    # ==================================================
    if user_query := st.chat_input(
        "Ask a question about your research papers..."
    ):

        # Save and display user message
        st.session_state.messages.append(
            {"role": "user", "content": user_query}
        )

        with st.chat_message("user"):
            st.markdown(user_query)

        # ==============================================
        # ASSISTANT RESPONSE
        # ==============================================
        with st.chat_message("assistant"):

            with st.spinner("Thinking..."):

                # --------------------------------------
                # RETRIEVE FROM EVERY UPLOADED PAPER
                # --------------------------------------
                retrieved_docs = []

                for filename in st.session_state.uploaded_filenames:

                    retriever = (
                        st.session_state.vector_store.as_retriever(
                            search_type="mmr",
                            search_kwargs={
                                "k": 4,
                                "fetch_k": 15,
                                "filter": {
                                    "filename": filename
                                }
                            }
                        )
                    )

                    docs = retriever.invoke(user_query)
                    retrieved_docs.extend(docs)

                # --------------------------------------
                # BUILD CONTEXT WITH SOURCE INFORMATION
                # --------------------------------------
                context_text = "\n\n".join(
                    [
                        f"Source File: {doc.metadata.get('filename', 'Unknown')}\n"
                        f"Page: {doc.metadata.get('page', 'Unknown')}\n"
                        f"Content:\n{doc.page_content}"
                        for doc in retrieved_docs
                    ]
                )

                # --------------------------------------
                # CREATE LLM
                # --------------------------------------
                llm = get_llm()

                # --------------------------------------
                # PROMPT
                # --------------------------------------
                prompt = ChatPromptTemplate.from_messages([
                    (
                        "system",
                        """You are a research paper assistant.

Answer the user's question using ONLY the provided context.

The context may contain information from multiple research papers.

If the user asks to compare two or more papers:
- Use information from EACH relevant paper in the context.
- Clearly identify each paper by its source filename.
- Compare their objectives, methods, results, or other requested aspects.
- Do not ignore a paper if relevant context from that paper is provided.

Do not make up information or use outside knowledge.
If the answer is not present in the context, clearly say that the
information is not available in the retrieved passages.

When possible, mention the source file and page number supporting
your answer.

Context:
{context}
"""
                    ),
                    (
                        "human",
                        "{question}"
                    ),
                ])

                # --------------------------------------
                # RUN LLM CHAIN
                # --------------------------------------
                chain = prompt | llm | StrOutputParser()
                try:

                    response = chain.invoke(
                        {
                            "context": context_text,
                            "question": user_query
                        }
                    )
                except Exception as e:
                     response = "Sorry, I couldn't generate a response at the moment. Please try again."
                     st.error(response)

                # Display answer
                st.markdown(response)

                # ======================================
                # DISPLAY RETRIEVED SOURCE CHUNKS
                # ======================================
                with st.expander("🔍 View Retrieved Source Chunks"):

                    if retrieved_docs:

                        for i, doc in enumerate(retrieved_docs, 1):

                            source = doc.metadata.get("filename", "Unknown")
                            page = doc.metadata.get("page", "Unknown")

                            st.markdown(
                                f"**Chunk {i} — File:** `{source}` (Page {page})"
                            )

                            preview = doc.page_content[:500]
                            if len(doc.page_content) > 500:
                                preview += "..."

                            st.text(preview)

                    else:
                        st.warning("No relevant chunks were retrieved.")

                

        # Save assistant response in chat history
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": response
            }
        )

# --------------------------------------------------
# BEFORE FILES ARE PROCESSED (Call-to-action view)
# --------------------------------------------------
else:
    st.markdown("---")
    st.info(
        "👆 Please upload your research PDFs and click **Process Papers** "
        "above to unlock structured extraction and chat intelligence."
    )