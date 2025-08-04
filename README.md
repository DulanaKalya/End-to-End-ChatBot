# 📚 RAG Q&A Conversation With PDF Uploads and Chat History

This project is an **end-to-end Conversational Retrieval-Augmented Generation (RAG)** system that allows users to upload **PDF documents** and ask questions based on the document content. The system remembers past interactions using **chat history**, improving multi-turn dialogue accuracy.

Built using **LangChain**, **Streamlit**, **Groq LLMs**, and **ChromaDB**, it showcases how to implement a document-aware, session-based chatbot using modern LLM infrastructure.

## 📁 Project Structure

```
End-to-End-ChatBot/
├── Ollama/
│   └── app.py                      # Optional: LLM local experimentation (e.g., Ollama)
│
├── RAG Document Q .../            # Another RAG-based PDF project (simplified)
│   ├── pdf/
│   └── app.py
│
├── RAG-Q-A-Conversation.../       # ✅ This is the main project
│   ├── .env                       # HuggingFace token
│   ├── .gitignore
│   ├── app.py                     # Main Streamlit app for RAG Q&A with chat history
│   ├── requirements.txt           # Required libraries
│   └── temp.pdf                   # Temporary storage for uploaded PDFs
│
├── README.md                      # 📘 Project documentation (you're writing this)
```

## 🚀 Features

* 🗂️ Upload one or multiple PDF files
* 🔍 Ask questions about the content in natural language
* 🧠 Uses Retrieval-Augmented Generation for accurate context-based answers
* 💬 Maintains **session-based chat history** using `RunnableWithMessageHistory`
* 📚 Reformulates questions based on chat history (history-aware retriever)
* 🤖 Powered by **Groq LLM (Gemma-2-9b-IT)** and **MiniLM embeddings**
* 🌐 Simple and interactive UI built using **Streamlit**

## 🛠️ Tech Stack & Tools

| Tool / Library | Purpose |
|---|---|
| `Streamlit` | Frontend UI |
| `LangChain` | RAG pipeline, prompts, history |
| `Groq` + `ChatGroq` | Fast LLM inference (Gemma 2 9B) |
| `HuggingFaceEmbeddings` | Embeddings via MiniLM |
| `Chroma` | In-memory vector database |
| `PyPDFLoader` | Load PDFs as LangChain docs |
| `RecursiveCharacterTextSplitter` | Chunking long documents |
| `dotenv` | API key management |

## 🧪 How It Works

1. **User uploads PDF(s)**
2. Documents are loaded and split into chunks
3. Embeddings are created using `all-MiniLM-L6-v2`
4. Chunks are stored in a Chroma vector store
5. Questions are reformulated using chat history (`history-aware retriever`)
6. Final answers are generated using the RAG pipeline
7. Chat history is saved session-wise

## 🖼️ Screenshots

*Add screenshots of the following:*

* 🏠 Initial interface (before PDF upload)
* 📄 After uploading a PDF
* 💬 Asking a question
* 📜 Displaying answer and chat history

You can add images like this:

```markdown
![Chat Interface](screenshots/chat_interface.png)
```

## 🔐 Environment Setup

1. Clone the repo:

```bash
git clone https://github.com/yourusername/your-repo-name.git
cd your-repo-name
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Create a `.env` file with your Hugging Face token:

```env
HF_TOKEN=your_huggingface_token
```

4. Run the app:

```bash
streamlit run RAG-Q-A-Conversation-With-PDF-Including-Chat-Historyapp.py
```

## 📌 Requirements

* Python 3.8+
* Streamlit
* LangChain
* Groq API Key (you will input it in the app)
* HuggingFace Token for embeddings

## 📈 Example Use Cases

* Ask questions from academic research papers
* Query legal documents or contracts
* Summarize multi-page policy documents
* Interactive reading assistant for students

## 🧠 Future Improvements

* Add source document references with answers
* Allow multiple session management
* Integrate other LLM providers (OpenAI, Anthropic, etc.)
* UI enhancements for chat layout

## 👨‍💻 Author

Built with ❤️ using modern LLM and RAG technologies.
