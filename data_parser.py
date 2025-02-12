import pandas as pd
import json
import re
import os
import io
from PIL import Image
import requests
from bs4 import BeautifulSoup


SINGLE_ARTICLES_FILENAME = 'data/single_articles.xlsx'
WEEKLY_ARTICLES_FILENAME = 'data/weekly_articles.xlsx'
ALL_ARTICLES_OUTPUT_FILENAME = 'data/all_articles_v2.xlsx'
IMAGES_DATA_DIR="data/imagesv2/"
IMAGES_DATASET_CONFIG_FILE="images_dataset_v2.json"


def parse_single_articles(input_filename: str):
    df = (
        pd.read_excel(input_filename)
        .drop(columns=['web-scraper-order', 'web-scraper-start-url', 'image-src'])
        .rename(columns={'thebatch_root-href': 'article_url'})
        .dropna()
        .fillna('[]')
        .replace('[]', None)
    )
    df['text'] = df.text.str.replace(u'\xa0', u' ')

    def clean_image(image: str) -> str:
        if image is not None:
            img_src = json.loads(image)[0]['image-src']
            img_src = img_src.replace('%3A', ':').replace('%2F', '/').strip('"')
            match_link = re.search(r'url=(https://[^&]+)', img_src)
            return [match_link.group(1)] if match_link else None

    df['image_cleaned'] = df['image'].apply(clean_image)

    df = df.drop(columns="image")
    df = df.rename(columns={"image_cleaned": "image"})

    return df[['article_url', 'text', 'image']]


def parse_weekly_articles(input_filename: str):
    df = (
        pd.read_excel(input_filename)
        .drop(columns=['web-scraper-order', 'web-scraper-start-url'])
        .rename(columns={'thebatch_root-href': 'article_url', 'text': 'raw_html'})
        .dropna()
    )

    def extract_articles(html: str):
        articles = []
        
        raw_articles = html.split('<hr>')
        for raw_article in raw_articles:
            article_soup = BeautifulSoup(raw_article, 'html.parser')
            text = article_soup.get_text(separator=' ', strip=True)
            images = [img['src'] for img in article_soup.find_all('img') if 'src' in img.attrs]
            articles.append({
                'text': text,
                'image': images
            })
        return articles

    df['articles'] = df.raw_html.apply(extract_articles)

    df_exploded = df.drop(columns='raw_html').explode('articles')
    df_exploded = df_exploded.reset_index()
    df_exploded = df_exploded.drop(columns="index")

    df_normalized = pd.json_normalize(df_exploded['articles'])
    full_df = pd.concat([df_exploded, df_normalized], axis=1)
    full_df = full_df[full_df.text.str.len() > 0].drop(columns=['articles'])

    return full_df


def parse_articles(
        single_filename: str = SINGLE_ARTICLES_FILENAME,
        weekly_filename: str = WEEKLY_ARTICLES_FILENAME,
        output_filename: str = ALL_ARTICLES_OUTPUT_FILENAME
    ):
    
    single_articles_df = parse_single_articles(single_filename)
    weekly_articles_df = parse_weekly_articles(weekly_filename)

    all_articles_df = pd.concat([single_articles_df, weekly_articles_df], ignore_index=True)
    all_articles_df[['article_url', 'text', 'image']].to_csv(output_filename, index=False)


def create_image_dataset_config(
        articles_filename: str = ALL_ARTICLES_OUTPUT_FILENAME, 
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

    os.makedirs(output_folder, exist_ok=True)

    with open(f'{output_folder}{img_config_file}', 'w') as file:
        json.dump(images, file)
    
    return images


def save_images_to_local(output_folder=IMAGES_DATA_DIR):
    
    img_dataset = create_image_dataset_config(
        ALL_ARTICLES_OUTPUT_FILENAME, 
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


parse_articles()
save_images_to_local()





