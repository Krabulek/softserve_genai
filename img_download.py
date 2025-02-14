import pandas as pd
import json
import os
import io
from PIL import Image
import requests
from dotenv import load_dotenv


load_dotenv(override=True)

ARTICLES_FILENAME = os.getenv("ARTICLES_FILENAME")
IMAGES_DATA_DIR = os.getenv("IMAGES_DATA_DIR")
IMAGES_DATASET_CONFIG_FILE = os.getenv("IMAGES_DATASET_CONFIG_FILE")

os.makedirs(IMAGES_DATA_DIR, exist_ok=True)


def create_image_dataset_config(
        articles_filename: str = ARTICLES_FILENAME, 
        output_folder: str = IMAGES_DATA_DIR,
        img_config_file:str = IMAGES_DATASET_CONFIG_FILE
    ):
    
    all_articles_df = pd.read_csv(articles_filename)
    all_articles_df.info()

    all_articles_df['image'] = all_articles_df['image'].str.strip('"').str.strip("[]").str.split(',')
    unique_images = all_articles_df.explode('image').image.dropna().unique()
    cleaned_image_urls = {el.replace("'", "") for el in unique_images if len(el) > 5}

    images = {}
    for el in cleaned_image_urls:
        if el is not None:
            image_name = el[51:].replace('/', '_').replace('.', '_')
            images[image_name] = el

    with open(f'{output_folder}{img_config_file}', 'w') as file:
        json.dump(images, file)
    
    return images


def save_images_locally(output_folder=IMAGES_DATA_DIR):
    
    img_dataset = create_image_dataset_config(
        ARTICLES_FILENAME, 
        output_folder,
        IMAGES_DATASET_CONFIG_FILE
    )

    for image_name in img_dataset:
        image_url = img_dataset[image_name]
        try:
            image = requests.get(image_url).content
            image = Image.open(io.BytesIO(image))

            image = image.resize((448, 448))

            output_path = os.path.join(output_folder, f"{image_name}.png")

            image.save(output_path, format='PNG')
            print(f"Image saved in: {output_path}")
        except Exception as e:
            print(e)


save_images_locally()
