import pandas as pd
import html 
import re

def preprocess_data(df):
    # first we filter the dataframe to be verified purchases only
    df = df[df['verified_purchase'] == True].copy()
    
    df = df[['rating', 'title', 'text']].copy()
    
    # fill missing titles with text and vice versa
    df['title'] = df.apply(lambda x: x['text'] if pd.isna(x['title']) and not pd.isna(x['text']) else x['title'], axis=1)
    df['text'] = df.apply(lambda x: x['title'] if pd.isna(x['text']) and not pd.isna(x['title']) else x['text'], axis=1)
    
    # if both title and text are missing fill based on rating
    missing_mask = df['title'].isna() & df['text'].isna()
    df.loc[missing_mask & (df['rating'].isin([4, 5])), ['title', 'text']] = 'very good'
    df.loc[missing_mask & (df['rating'] == 3), ['title', 'text']] = 'good'
    df.loc[missing_mask & (df['rating'].isin([1, 2])), ['title', 'text']] = 'bad'
    
    # clean the titles and text
    df['title'] = df['title'].apply(clean_text)
    df['text'] = df['text'].apply(clean_text)
    
    # setup columns with features we want
    df['title_exclamations'] = df['title'].apply(lambda x: x.count('!') if pd.notna(x) else 0)
    df['title_questions'] = df['title'].apply(lambda x: x.count('?') if pd.notna(x) else 0)
    df['title_word_count'] = df['title'].apply(lambda x: len(x.split()) if pd.notna(x) else 0)
    
    df['text_exclamations'] = df['text'].apply(lambda x: x.count('!') if pd.notna(x) else 0)
    df['text_questions'] = df['text'].apply(lambda x: x.count('?') if pd.notna(x) else 0)
    df['text_word_count'] = df['text'].apply(lambda x: len(x.split()) if pd.notna(x) else 0)
    
    # >= 3 is positive rating, otherwise negative
    df['positive_review'] = (df['rating'] >= 3).astype(int)
    
    return df

def clean_text(text):
    if pd.isna(text):
        return text
    
    text = str(text)
    
    # Remove any thml entities
    text = html.unescape(text)
    
    # Remove an html tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Remove emojis
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"
        u"\U0001F300-\U0001F5FF"
        u"\U0001F680-\U0001F6FF"
        u"\U0001F1E0-\U0001F1FF"
        "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r'', text)
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove extra spaces
    text = ' '.join(text.split())
    
    return text

