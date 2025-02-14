import pandas as pd
import json
import re
import os
from bs4 import BeautifulSoup
from dotenv import load_dotenv


load_dotenv(override=True)

SINGLE_ARTICLES_FILENAME = os.getenv("SINGLE_ARTICLES_FILENAME")
WEEKLY_ARTICLES_FILENAME = os.getenv("WEEKLY_ARTICLES_FILENAME")
ARTICLES_FILENAME = os.getenv("ARTICLES_FILENAME")


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
        output_filename: str = ARTICLES_FILENAME
    ):
    
    single_articles_df = parse_single_articles(single_filename)
    weekly_articles_df = parse_weekly_articles(weekly_filename)

    all_articles_df = pd.concat([single_articles_df, weekly_articles_df], ignore_index=True)
    all_articles_df[['article_url', 'text', 'image']].to_csv(output_filename, index=False)


parse_articles()
