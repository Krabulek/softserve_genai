import sys
import os
import ssl
import certifi
from glob import glob
from tqdm import tqdm

from encoder import emb_text, emb_image
from milvus_utils import get_milvus_client, create_text_collection, create_image_collection

from dotenv import load_dotenv


load_dotenv()
TEXT_COLLECTION_NAME = os.getenv("TEXT_COLLECTION_NAME")
IMAGE_COLLECTION_NAME = os.getenv("IMAGE_COLLECTION_NAME")
MILVUS_ENDPOINT = os.getenv("MILVUS_ENDPOINT")


def get_articles(data_dir):
    """Load documents and split each into chunks.

    Return:
        A dictionary of text chunks with the filepath as key value.
    """
    text_dict = {}
    for file_path in glob(os.path.join(data_dir, "**/*.md"), recursive=True):
        if file_path.endswith(".md"):
            with open(file_path, "r") as file:
                file_text = file.read().strip()
            text_dict[file_path] = file_text.split("# ")
    return text_dict


def get_images(data_dir):
    """
    Return:
        A list of all file names for images.
    """
    image_list = glob(os.path.join(data_dir, "images", "*.png"))
    return image_list


milvus_client = get_milvus_client(uri=MILVUS_ENDPOINT, token=None)

ssl_context = ssl.create_default_context(cafile=certifi.where())

data_dir = sys.argv[-1]


# TEXT COLLECTION

text_dict = get_articles(data_dir)

dim = len(emb_text("test"))
create_text_collection(milvus_client=milvus_client, collection_name=TEXT_COLLECTION_NAME, dim=dim, drop_old=False)

data = []
count = 0
for i, filepath in enumerate(tqdm(text_dict, desc="Creating embeddings")):
    chunks = text_dict[filepath]
    for line in chunks:
        try:
            vector = emb_text(line)
            data.append({"vector": vector, "text": line})
            count += 1
        except Exception as e:
            print(
                f"Skipping file: {filepath} due to an error occurs during the embedding process:\n{e}"
            )
            continue
print("Total number of loaded documents:", count)

mr = milvus_client.insert(collection_name=TEXT_COLLECTION_NAME, data=data)
print("Total number of entities/chunks inserted:", mr["insert_count"])


# IMAGE COLLECTION

image_list = get_images(data_dir)

dim = len(emb_image(image_list[0]))
create_text_collection(milvus_client=milvus_client, collection_name=TEXT_COLLECTION_NAME, dim=dim, drop_old=False)

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
print("Total number of entities/chunks inserted:", mr["insert_count"])