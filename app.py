from flask import Flask, render_template, request
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import system_prompt
import os

os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['OMP_NUM_THREADS'] = '1'

# Initialize Flask app
app = Flask(__name__)

# Load environment variables
load_dotenv()

# Get API keys from environment
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

# Set environment variables
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

print("🚀 Loading embeddings model...")
embeddings = download_hugging_face_embeddings()

print("🔗 Connecting to Pinecone...")
index_name = "healthmate-ai"
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

print("⚙️ Setting up retriever and RAG chain...")
retriever = docsearch.as_retriever(
    search_type="similarity", search_kwargs={"k": 3})

chatModel = ChatOpenAI(model="gpt-4o")
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(chatModel, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

print("✅ HealthMate-AI is ready!")


@app.route("/")
def index():
    """Render the main chat interface"""
    return render_template('chat.html')


@app.route("/get", methods=["POST"])
def chat():
    """Handle chat messages"""
    try:
        msg = request.form.get("msg", "")
        if not msg:
            return "Please provide a message", 400

        print(f"📝 User query: {msg}")
        response = rag_chain.invoke({"input": msg})
        answer = response["answer"]
        print(f"💬 Response: {answer}")

        return str(answer)
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return f"I apologize, but I encountered an error processing your request. Please try again.", 500


@app.route("/health")
def health():
    """Health check endpoint for monitoring"""
    return {"status": "healthy", "service": "HealthMate-AI"}, 200


if __name__ == '__main__':
    # For local development
    app.run(host="0.0.0.0", port=8080, debug=False)
