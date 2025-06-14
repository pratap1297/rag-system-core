from setuptools import setup, find_packages

setup(
    name="rag-system-core",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "fastapi",
        "uvicorn",
        "python-multipart",
        "PyPDF2",
        "python-docx",
        "openpyxl",
        "pdf2image",
        "pytesseract",
        "azure-cognitiveservices-vision-computervision",
        "python-dotenv",
        "nltk",
        "sentence-transformers",
        "faiss-cpu",
        "torch",
        "scikit-learn",
        "numpy",
        "Pillow"
    ],
    python_requires=">=3.8",
) 