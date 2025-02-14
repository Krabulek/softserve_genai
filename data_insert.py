import pandas as pd
import os
import ssl
import certifi
from glob import glob
from tqdm import tqdm

from rag_app.encoder import emb_text, emb_image
from rag_app.milvus_utils import get_milvus_client, create_text_collection, create_image_collection

from dotenv import load_dotenv


load_dotenv(override=True)

TEXT_COLLECTION_NAME = os.getenv("TEXT_COLLECTION_NAME")
IMAGE_COLLECTION_NAME = os.getenv("IMAGE_COLLECTION_NAME")
MILVUS_ENDPOINT = os.getenv("MILVUS_ENDPOINT")
ARTICLES_FILENAME = os.getenv("ARTICLES_FILENAME")
IMAGES_DATA_DIR = os.getenv("IMAGES_DATA_DIR")

milvus_client = get_milvus_client(uri=MILVUS_ENDPOINT, token=None)

ssl_context = ssl.create_default_context(cafile=certifi.where())

print(f"Milvus DB: {MILVUS_ENDPOINT}, Text Collection: {TEXT_COLLECTION_NAME}, Image Collection: {IMAGE_COLLECTION_NAME}")


def get_articles(articles_filename: str = ARTICLES_FILENAME):
    input_filename = articles_filename
    return pd.read_csv(input_filename)


def get_images(img_data_dir: str = IMAGES_DATA_DIR):
    image_list = glob(os.path.join(img_data_dir, "*.png"))
    return image_list


# TEXT COLLECTION

text_df = get_articles()

dim = len(emb_text("test"))
create_text_collection(
    milvus_client=milvus_client,
    collection_name=TEXT_COLLECTION_NAME,
    dim=dim,
    drop_old=True
)

print("Generating text embeddings")
try:
    doc_embeddings = emb_text(text_df.text)
except Exception as e:
    print(
        f"Failed to generate embeddings:\n{e}"
    )

data = []
for index, row in text_df.iterrows():
    data.append({
        "vector": doc_embeddings[index],
        "text": row.text,
        "article_url": row.article_url, 
        "image_url": row.image
        })
print("Total number of loaded documents:", len(data))

mr = milvus_client.insert(collection_name=TEXT_COLLECTION_NAME, data=data)
print("Total number of inserted documents:", mr["insert_count"])


# IMAGE COLLECTION

image_list = get_images()

dim = len(emb_image(image_list[0]))
create_image_collection(
    milvus_client=milvus_client,
    collection_name=IMAGE_COLLECTION_NAME,
    dim=dim,
    drop_old=True
)

image_dict = {}
for image_path in tqdm(image_list, desc="Generating image embeddings: "):
    try:
        image_dict[image_path] = emb_image(image_path)
    except Exception as e:
        print(f"Failed to generate embedding for {image_path}. Skipped.")
        continue
print("Number of encoded images:", len(image_dict))

mr =milvus_client.insert(
    collection_name=IMAGE_COLLECTION_NAME,
    data=[{"image_path": k, "vector": v} for k, v in image_dict.items()],
)
print("Total number of images inserted:", mr["insert_count"])