# Axis Guidelines AI Assistant

An intelligent AI-powered assistant that helps users quickly find answers from company guidelines using advanced natural language processing and vector search technology.

## 🚀 Features

- **Instant Answers**: Get accurate answers from company guidelines in seconds
- **AI-Powered**: Uses Google Gemini 2.0 Flash for intelligent responses
- **PDF Processing**: Automatically processes and indexes PDF documents
- **Modern UI**: Beautiful, responsive Streamlit interface with custom styling
- **Vector Search**: Fast similarity search using FAISS and sentence transformers
- **Caching**: Optimized performance with intelligent caching system

## 📋 Prerequisites

- Python 3.8 or higher
- Google AI API key (Gemini API)
- PDF documents to process

## 🏗️ Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/okimadesigner/axis-guidelines.git
   cd axis-guidelines
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up your API key**
   - Create a `api_key.txt` file (for processing) and add your Google AI API key
   - For Streamlit deployment, add `GOOGLE_API_KEY` to your secrets.toml file

## 🔧 Usage

### 1. Process Documents

Add your PDF documents to the `test_pdfs/` folder, then run:

```bash
python processor.py
```

This will:
- Extract text from all PDFs
- Split content into meaningful chunks
- Create vector embeddings
- Build a FAISS search index
- Save processed data to `chunks.pkl` and `index.faiss`

### 2. Launch the Web App

```bash
streamlit run app.py
```

The app will be available at `http://localhost:8501`

### 3. Ask Questions

- Type questions about your guidelines in the text input
- Click "Get Answer" or press Enter
- View AI-generated answers with source references
- Browse recent questions in the chat history

## 📁 Project Structure

```
axis-guidelines/
├── app.py                 # Main Streamlit application
├── processor.py           # PDF processing and indexing script
├── requirements.txt       # Python dependencies
├── chunks.pkl            # Processed text chunks (generated)
├── index.faiss           # FAISS vector search index (generated)
├── .gitignore            # Git ignore rules
├── test_pdfs/            # Directory for PDF documents
│   ├── content_design_guidelines_1.0.1.pdf
│   └── marketing_brand_guidelines_1.0.1.pdf
└── README.md             # This file
```

## 🔧 Configuration

### Environment Variables

For production deployment, set these environment variables:

```bash
GOOGLE_API_KEY=your_api_key_here
```

### Streamlit Configuration

For better performance, you can configure Streamlit by creating a `.streamlit/config.toml`:

```toml
[server]
headless = true
port = 8501

[browser]
gatherUsageStats = false
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📊 Technical Details

- **Language Model**: Google Gemini 2.0 Flash Experimental
- **Embedding Model**: sentence-transformers/all-MiniLM-L6-v2
- **Vector Search**: FAISS (Facebook AI Similarity Search)
- **UI Framework**: Streamlit with custom CSS
- **Chunk Size**: 500 characters with 400-character overlap

## 🔄 Updating Documents

To add new guidelines or update existing ones:

1. Add/remove PDF files in the `test_pdfs/` directory
2. Run `python processor.py` to reprocess documents
3. Commit and push the updated `chunks.pkl` and `index.faiss` files

## 🐛 Troubleshooting

### Common Issues

1. **API Key Error**: Ensure your Google AI API key is correctly configured
2. **Missing Files**: Run `processor.py` if `chunks.pkl` or `index.faiss` are missing
3. **Memory Issues**: The app includes automatic cache management for performance

### Contact

For issues or feature requests, please contact:
- Email: subzero@freecharge.com

## 📝 License

This project is proprietary software. All rights reserved.

## 🕒 Last Updated

October 2025

---

Built with ❤️ for the Axis team using cutting-edge AI technology.
