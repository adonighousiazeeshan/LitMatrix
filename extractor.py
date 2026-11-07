#Handles the Pydantic schema and structured extraction logic.
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()


# Define Pydantic Schema for Feature 3: Structured Research Extraction
class ResearchExtraction(BaseModel):
  algorithm: str = Field(
      description="The primary algorithm or model architecture used in the study."
  )
  dataset: str = Field(
      description=(
          "The dataset or benchmarks used for evaluation (or 'Not reported')."
      )
  )
  metrics: str = Field(
      description=(
          "The evaluation metrics reported, such as Accuracy, F1, RMSE (or 'Not"
          " reported')."
      )
  )
  limitations: str = Field(
      description=(
          "Limitations or future work noted by the authors (or 'Not reported')."
      )
  )


def extract_paper_metadata(vector_store, uploaded_filenames):
  """Extracts standardized research details across uploaded documents."""
  llm = ChatGroq(model_name="openai/gpt-oss-20b")
  structured_llm = llm.with_structured_output(ResearchExtraction)

  extraction_data = []
 

  for filename in uploaded_filenames:
    retriever=vector_store.as_retriever(search_type="mmr",search_kwargs={"k":6,"fetch_k":25,"filter":{"filename":filename}})
    query = f"What are the algorithm, dataset, evaluation metrics, and limitations reported in {filename}?"
    retrieved_docs = retriever.invoke(query)
    context_text = "\n\n".join([ f"Source: {doc.metadata.get('source', 'Unknown')}\n"
        f"Page: {doc.metadata.get('page', 'Unknown')}\n"
        f"Evidence:\n{doc.page_content}"
        for doc in retrieved_docs])

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            (
                "Extract the requested research details accurately from the"
                " provided context. If a field is missing, state 'Not"
                " reported'."
            ),
        ),
        ("human", "Context:\n{context}\n\nExtract the structured details.")
    ])

    chain = prompt | structured_llm
    try:
      result = chain.invoke({"context": context_text})
      extraction_data.append({
          "Paper": filename,
          "Algorithm / Model": result.algorithm,
          "Dataset": result.dataset,
          "Metrics": result.metrics,
          "Limitations": result.limitations,
      })
    except Exception as e:
      extraction_data.append({
          "Paper": filename,
          "Algorithm / Model": "Extraction failed",
          "Dataset": str(e),
          "Metrics": "-",
          "Limitations": "-",
      })

  return extraction_data