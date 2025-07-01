from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
import gradio as gr
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
 
load_dotenv()
 
# Configuration
DATA_PATH = "data"
CHROMA_PATH = "chroma_db"
 
# Initialize models
embeddings_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
llm = ChatGoogleGenerativeAI(temperature=0.5, model="gemini-1.5-flash")
 
# ChromaDB setup
vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings_model,
    persist_directory=CHROMA_PATH,
)
retriever = vector_store.as_retriever(search_kwargs={"k": 5})
 
def stream_response(message, history):
    # Retrieve relevant documents
    docs = retriever.invoke(message)
    knowledge = "\n\n".join(doc.page_content for doc in docs)
    # Convert history to LangChain format
    messages = []
    for human_msg, ai_msg in history:
        messages.extend([
            HumanMessage(content=human_msg),
            AIMessage(content=ai_msg)
        ])
    messages.append(HumanMessage(content=message))
 
    # HYBRID SYSTEM PROMPT WITH FALLBACK LOGIC
    system_prompt = f"""
    You are a helpful assistant. Answer questions using this priority system:
    1. PRIMARY SOURCE: Use the provided knowledge when relevant.
    2. SECONDARY SOURCE: If knowledge is missing/irrelevant, use your internal knowledge.
    3. FALLBACK: If unsure, say you don't know.
 
    Always be concise and factual.
 
    PROVIDED KNOWLEDGE (may be empty):
    {knowledge if knowledge else "No relevant documents found."}
    """
 
    full_messages = [HumanMessage(content=system_prompt)] + messages
 
    # Stream response
    partial_message = ""
    try:
        for chunk in llm.stream(full_messages):
            partial_message += chunk.content
            yield partial_message
    except Exception as e:
        yield f"Error: {str(e)}"
 
# Gradio interface
chatbot = gr.ChatInterface(
    stream_response,
    textbox=gr.Textbox(
        placeholder="Ask me anything...",
        container=False,
        autoscroll=True,
        scale=7
    )
)
chatbot.launch()