
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from typing import List
from dotenv import load_dotenv

load_dotenv()


class ClaimEvaluation(BaseModel):
    claim: str = Field(
        description="A single distinct factual statement extracted from the response."
    )

    is_supported: bool = Field(
        description="True if directly supported by the retrieved context, False otherwise."
    )

    reasoning: str = Field(
        description="Brief explanation based on the source text."
    )


class VerificationReport(BaseModel):
    evaluations: List[ClaimEvaluation] = Field(
        description="At most 4 important factual claims evaluated from the response.",
        max_length=4
    )

    overall_trust_score: int = Field(
        ge=0,
        le=100,
        description="Whole-number percentage of supported claims from 0 to 100."
    )


def verify_claim(response_text: str, retrieved_doc: list):

    context = "\n\n".join([
        f"Source ({doc.metadata.get('filename', 'unknown')}): "
        f"{doc.page_content}"
        for doc in retrieved_doc
    ])

    llm = ChatGroq(
        model_name="openai/gpt-oss-20b",
        temperature=0,
        max_tokens=2048
    )

    structured_llm = llm.with_structured_output(
        VerificationReport,
        method="function_calling"
    )

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """
            You are an expert AI research auditor and fact-checker.

            Compare the AI response against the provided source passages.

            Instructions:
            1. Extract at most 4 important factual claims.
            2. Evaluate each claim independently.
            3. Mark a claim supported only when the source directly supports it.
            4. If evidence is insufficient, mark it unsupported.
            5. Keep reasoning concise (one or two sentences).
            6. Calculate overall_trust_score as:

               (number of supported claims / total claims) * 100

            7. Return a complete structured response.
            8. Do not generate incomplete evaluations.
            """
        ),
        (
            "human",
            """
            Source Passages:
            {context}

            AI Response to Audit:
            {response}
            """
        )
    ])

    chain = prompt | structured_llm

    report = chain.invoke({
        "context": context,
        "response": response_text
    })

    return report
