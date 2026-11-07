from pydantic import BaseModel,Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from typing import List
from dotenv import load_dotenv

load_dotenv()

class ClaimEvaluation(BaseModel):
    model_config = {"extra": "allow"}
    claim:str=Field(description="A single distinct factual statement extracted from the response.")
    is_supported:bool=Field(description="True if the claim is directly supported by the retrieved context, False otherwise.")
    reasoning:str=Field(description="Brief explanation of why the claim is supported or unsupported based on the source text.")


class verificationreport(BaseModel):
    model_config = {"extra": "allow"}
    evaluations:List[ClaimEvaluation]=Field(description=  "A maximum of 6 important factual claims "
        "evaluated from the response.")
    overall_trust_score:int=Field(description="Whole-number percentage of supported claims from 0 to 100.")

def verify_claim(response_text:str,retrieved_doc:list):
    context='\n\n'.join([ f"source ({doc.metadata.get('filename','unknown')}):{doc.page_content}"for doc in retrieved_doc])
    llm=ChatGroq(model_name="openai/gpt-oss-20b", temperature=0.0)
    structured_llm=llm.with_structured_output(verificationreport)
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are an expert AI research auditor and fact-checker.
            Analyze the AI response and cross-check every factual claim against
            the provided source passages.

            For each claim, determine whether it is directly supported by the
            retrieved context. If the context does not contain enough evidence,
            mark it as unsupported.
            """
        ),
        (
            "human",
            "Source Passages:\n{context}\n\nAI Response to Audit:\n{response}"
        )
    ])
    chain=prompt|structured_llm
    report=chain.invoke({'context':context,"response":response_text})
    return report

