from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough 

import os 
from dotenv import load_dotenv  #Dotenv is a lightweight library used to load environment variables from a .env file into your application's runtime environment
load_dotenv()

pdf_path = r"C:\Users\HP\Desktop\GEN_AI\AI-Course-Certification-RAG\data\courses_certifications.pdf"
docs = PyPDFLoader(pdf_path).load()
print("\nUploaded PDF name:", os.path.basename(pdf_path)) #os.path.basename(pdf_path)-takes the last part of the path
print("\nTotal number of pages in the document:", len(docs))
print("\nTotal number of characters in the document:", sum(len(d.page_content) for d in docs))

#print("Uploaded PDF name:", os.path.basename(pdf_path)) 
#print()
#print("Total number of pages in the document: ",len(docs))
#print()
#print("Total number of characters in the document: ",sum(len(d.page_content) for d in docs))

splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=0)
chunks = splitter.split_documents(docs)
print("\nTotal number of chunks created: ",len(chunks))

embedding = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
vectorstore = FAISS.from_documents(chunks, embedding)
print("\nvectore store created successfully")

retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
found = retriever.invoke("What are the courses and certifications offered by Google?")
print("\nRetrived chunks from the vector store based on the query: ")
for d in found:
    print("\n",d.page_content[:50])
    print("_"*60)

print("\nRetrived chunk from vector store with score:")
for d, score in vectorstore.similarity_search_with_score("What are the courses and certifications offered by Google?", k=3):
    print(f"Distance: {score:.4f} -> {d.page_content[:50]}...")




llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", temperature=0)

prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """Answer the user's question using only the information provided in the document context. If the answer is not available in the provided context, reply exactly:
             'I could not find this in the document.'
             Do not guess, assume, or use outside knowledge. Give a complete and clear answer using all information from the context that is directly relevant to the question.
             If the context contains conditions, exceptions, requirements, eligibility criteria, time periods, fees, approvals, procedures, or other details that affect the answer, include them.
             Do not include unrelated information."""
        ),
        (
            "human",
            "Context from the document: \n{context}\n\n User's question: {question}"
        )
])

def format_doc(docs):
    return "\n\n".join(d.page_content for d in docs)

rag_chain = (
    {
        "context": retriever | format_doc,
        "question": RunnablePassthrough(),
    }
    | prompt
    | llm
    | StrOutputParser()
)

print("\n\nRAG chain created successfully")

print("\n-------Test_1-------")
print(rag_chain.invoke("What are the courses and certifications offered by Google?"))

print("\n-------Test_2-------")
print(rag_chain.invoke("What are the courses and certifications offered by Microsoft?"))

print("\n-------Test_3-------")
print(rag_chain.invoke("What is the salary of someone who completes these certifications?"))

print("\n-------Test_4-------")
print(rag_chain.invoke("Which certification would be more suitable for someone who wants to enter cloud-focused careers without a programming-heavy start, and why?"))

