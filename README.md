# **Chat with PDF**  
An interactive app built using **Streamlit** and **Langchain** that allows you to upload PDF files and chat with them to extract information and answer questions.

---

## 🚀 **Features**  
✅ Upload multiple PDF files  
✅ Extract text from PDFs  
✅ Use AI to answer questions based on the document's content  
✅ Stores embeddings using **FAISS** for fast similarity search  
✅ Provides accurate responses using **Google Gemini**  

---

## 🛠️ **Tech Stack**  
- Python  
- Streamlit  
- Langchain  
- PyPDF2  
- FAISS  
- Google Gemini API  

---

## 📥 **Installation**  
1. Clone the repository:  
```bash
git clone https://github.com/Shabari-K-S/Chat-with-PDF.git
```

2. Navigate to the project folder:  
```bash
cd Chat-with-PDF
```

3. Create a virtual environment:  
```bash
python -m venv venv
source venv/bin/activate   # For Linux/Mac
venv\Scripts\activate      # For Windows
```

4. Install dependencies:  
```bash
pip install -r requirements.txt
```

5. Create a `.env` file and add your **Google API Key**:  
```
GOOGLE_API_KEY=your-google-api-key
```

---

## ▶️ **Usage**  
1. Run the app:  
```bash
streamlit run app.py
```

2. Upload PDF files using the sidebar.  
3. Once processing is complete, ask questions related to the uploaded documents.  

---

## 📝 **How It Works**  
1. **PDF Processing** – Extracts text from uploaded PDFs using `PyPDF2`.  
2. **Chunking** – Splits text into manageable chunks using `Langchain`.  
3. **Embedding and Vector Storage** – Converts text into embeddings and stores them in **FAISS** for fast search.  
4. **Question Answering** – Uses **Google Gemini** to generate answers based on the retrieved context.  

