# Mutlimodal RAG

### Preparation

> Prerequisites: Python 3.12 or higher

**1. Download Code and Visualized-BGE Model**
```bash
$ git clone https://github.com/FlagOpen/FlagEmbedding.git
$ !curl -O https://huggingface.co/BAAI/bge-visualized/resolve/main/Visualized_base_en_v1.5.pth
```

**2. Set Environment**

- Install dependencies

```bash
  $ pip install -r requirements.txt
  $ pip install -e FlagEmbedding
```

- Set environment variables

  Create the environment file [.env](./.env) with the following contents, inserting your Gemini API key:

  ```bash
  GEMINI_API_KEY=<your_api_key>
  TEXT_COLLECTION_NAME=the_batch_text
  IMAGE_COLLECTION_NAME=the_batch_image
  MILVUS_ENDPOINT=./the_batch.db
  SINGLE_ARTICLES_FILENAME=./input_data/single_articles.xlsx
  WEEKLY_ARTICLES_FILENAME=./input_data/weekly_articles.xlsx
  ARTICLES_FILENAME=./data/all_articles.csv
  IMAGES_DATA_DIR=./data/images/
  IMAGES_DATASET_CONFIG_FILE=images_dataset.json
  ```

## Parsing Article Data

Repository already contains the raw scrapped data in the *[input_data](./input_data)* directory.

**1. Clean the scrapped data**
```python
python3 data_parse.py
```

**2. Download images**
```python
python3 img_download.py
```

**3. Create Vector Database**

```python
python3 data_insert.py
```

## Running Streamlit application

```bash
streamlit run rag_app/app.py
```

You can now view your Streamlit app in your browser.

Local URL: http://localhost:8501
