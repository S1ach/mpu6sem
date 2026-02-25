import pandas as pd
import re
from tqdm import tqdm
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer


# Загрузка необходимых ресурсов NLTK
def download_nltk_resources():
    """Проверка и загрузка ресурсов NLTK при необходимости"""
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        nltk.download('punkt_tab')

    try:
        stopwords.words('english')
    except LookupError:
        nltk.download('stopwords')

    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet')


download_nltk_resources()

# Инициализация стоп-слов и лемматизатора
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


def preprocess_review(text):
    """
    Предобработка текста отзыва:
    - удаление всех символов, кроме букв
    - приведение к нижнему регистру
    - удаление лишних пробелов
    - токенизация
    - удаление стоп-слов
    - лемматизация
    """
    # Очистка текста
    text = re.sub(r'[^a-zA-Z]', ' ', str(text).lower())
    text = ' '.join(text.split())

    # Токенизация
    tokens = word_tokenize(text)

    # Лемматизация и удаление стоп-слов
    processed_tokens = [
        lemmatizer.lemmatize(word)
        for word in tokens
        if word not in stop_words
    ]

    return " ".join(processed_tokens)


# Активация прогресс-бара для pandas
tqdm.pandas()


def main():
    """Основная функция обработки данных"""
    # 1. Загрузка набора данных
    print("Загрузка данных...")
    df = pd.read_csv('reviews.csv')

    # 2. Предобработка текстов отзывов
    print("Предобработка отзывов...")
    df['review'] = df['review'].progress_apply(preprocess_review)

    # 3. Преобразование сентимента в бинарные метки
    print("Преобразование меток...")
    df['label'] = df['sentiment'].progress_apply(
        lambda x: 1 if x == 'positive' else 0
    )

    # 4. Сохранение обработанных данных
    print("Сохранение результатов...")
    df[['review', 'label']].to_csv('reviews_preprocessed.csv', index=False)
    print("Готово! Обработанные данные сохранены в 'reviews_preprocessed.csv'")


if __name__ == "__main__":
    main()