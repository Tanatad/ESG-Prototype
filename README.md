# Project Setup Guide

## Environment Setup

### Create Environment

```bash
python -m venv .venv
```

### Activate Environment

```bash
source .venv/bin/activate
```

## Install Requirements

Execute the following command after creating the environment to install the necessary packages:

```bash
pip install -r requirement-win.txt
```

If you install any new packages, update the `requirements.txt` file by running:

```bash
pip freeze > requirements.txt
```

## Additional Setup for PDF Files with Unstructured

### Requirements for MacOS

Ensure you have Homebrew installed and run:

```bash
brew install poppler tesseract libmagic
pip install python-magic
```

### Requirements for Linux

Run the following commands:

```bash
sudo apt-get install poppler-utils tesseract-ocr libmagic-dev
pip install python-magic
```

### Requirements for Windows

1. **Poppler**  
   Download the latest Poppler version from the [release page](https://github.com/oschwartz10612/poppler-windows) and install it. Ensure the `bin` folder is added to your system path.

2. **Tesseract-OCR**  
   Download the latest Tesseract-OCR version from the [release page](https://github.com/tesseract-ocr/tesseract/releases/tag/5.5.0) and install it. Again, add the `Tesseract-OCR` folder to your system path.

   **_Note_**: You must know how to install software and add paths to the system. For guidance on adding a path on Windows 10, refer to [this tutorial](https://www.architectryan.com/2018/03/17/add-to-the-path-on-windows-10/).

3. After these installations, run:
   ```bash
   pip install python-magic-bin
   ```

## Run the Application

Execute the following to run the application:

```bash
python -m app.main
```

## Developer Notes

- It is recommended to use Neo4j Cloud for this application.
- MongoDB can be used locally.

.\.myvenv_py312\Scripts\activate

MATCH (n) DETACH DELETE n;
DROP INDEX entity_description_embedding_index IF EXISTS;
DROP INDEX standard_chunk_embedding_index IF EXISTS;
DROP CONSTRAINT standard_chunk_chunk_id IF EXISTS;
DROP CONSTRAINT standard_document_doc_id IF EXISTS;
DROP CONSTRAINT constraint_907a464e IF EXISTS;

streamlit run streamlit_app.py
