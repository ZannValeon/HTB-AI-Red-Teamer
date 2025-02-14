import requests
import zipfile
import io
import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import joblib
import sys
import json
import joblib

def download():
    url = "https://academy.hackthebox.com/storage/modules/292/skills_assessment_data.zip"
    response = requests.get(url)
    if response.status_code == 200:
        print("Download successful")
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            z.extractall("skills_assessment_data")
            print("Extraction successful")
    else:
        print("Failed to download the dataset")

def dataset():
    df = pd.read_json("skills_assessment_data/train.json", orient="records")
    df.info()
    # Drop duplicates
    df = df.drop_duplicates()
    return df

def clean_text(text):
    # Remove HTML tags
    text = re.sub(r"<.*?>", " ", text)
    # Remove non-word characters (punctuation, etc.) but keep spaces
    text = re.sub(r"[^\w\s]", " ", text)
    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text

def preprocessing(df):
    # Basic text cleaning
    df["text"] = df["text"].apply(lambda x: x.lower())
    df["text"] = df["text"].apply(clean_text)
    return df

def train_model(df):
    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=0.3, random_state=42
    )

    # Create the pipeline
    pipeline = Pipeline([
        ("vectorizer", CountVectorizer(
            lowercase=True,
            stop_words="english",
            token_pattern=r"\b\w+\b",
            ngram_range=(1, 2)
        )),
        ("classifier", MultinomialNB())
    ])

    print("Training model...")
    pipeline.fit(X_train, y_train)
    print("Training complete!")

    # Save the trained model
    model_filename = "assessment.joblib"
    joblib.dump(pipeline, model_filename)
    print(f"Model saved to {model_filename}")

    return pipeline

def evaluate_model(model, new_texts):
    print("\nEvaluating new texts:")
    predictions = model.predict(new_texts)
    probabilities = model.predict_proba(new_texts)
    
    for text, pred, prob in zip(new_texts, predictions, probabilities):
        pred_label = "Good" if pred == 1 else "Bad"
        print(f"Text: {text[:60]}...")
        print(f"  -> Prediction: {pred_label} | Probabilities: {prob}")


def upload_model(pipeline):
    target = sys.argv[1]
    url = f'http://{target}:5000/api/upload'

    model_file_path = 'assessment.joblib'
    with open(model_file_path, "rb") as model_file:
        files = {"model": model_file}
        response = requests.post(url, files=files)

    # Pretty print the response from the server
    print(json.dumps(response.json(), indent=4))

if __name__ == "__main__":

    # Check for usage
    if len(sys.argv) < 2:
        print(f'Usage: {sys.argv[0]} <target_ip>')
        sys.exit(1)

    target = sys.argv[1]

    download()
    df = dataset()
    df = preprocessing(df)

    # Train the model
    model = train_model(df)

    # Example new texts
    new_texts = [
        "I went and saw this movie last night after being coaxed to by a few friends of mine. I'll admit that I was reluctant to see it because from what I knew of Ashton Kutcher he was only able to do comedy. I was wrong. Kutcher played the character of Jake Fischer very well, and Kevin Costner played Ben Randall with such professionalism. The sign of a good movie is that it can toy with our emotions. This one did exactly that. The entire theater (which was sold out) was overcome by laughter during the first half of the movie, and were moved to tears during the second half. While exiting the theater I not only saw many women in tears, but many full grown men as well, trying desperately not to let anyone see them crying. This movie was great, and I suggest that you go see it before you judge.",
        "As a recreational golfer with some knowledge of the sport's history, I was pleased with Disney's sensitivity to the issues of class in golf in the early twentieth century. The movie depicted well the psychological battles that Harry Vardon fought within himself, from his childhood trauma of being evicted to his own inability to break that glass ceiling that prevents him from being accepted as an equal in English golf society. Likewise, the young Ouimet goes through his own class struggles, being a mere caddie in the eyes of the upper crust Americans who scoff at his attempts to rise above his standing. <br /><br />What I loved best, however, is how this theme of class is manifested in the characters of Ouimet's parents. His father is a working-class drone who sees the value of hard work but is intimidated by the upper class; his mother, however, recognizes her son's talent and desire and encourages him to pursue his dream of competing against those who think he is inferior.<br /><br />Finally, the golf scenes are well photographed. Although the course used in the movie was not the actual site of the historical tournament, the little liberties taken by Disney do not detract from the beauty of the film. There's one little Disney moment at the pool table; otherwise, the viewer does not really think Disney. The ending, as in \"Miracle,\" is not some Disney creation, but one that only human history could have written.",
        "Bill Paxton has taken the true story of the 1913 US golf open and made a film that is about much more than an extra-ordinary game of golf. The film also deals directly with the class tensions of the early twentieth century and touches upon the profound anti-Catholic prejudices of both the British and American establishments. But at heart the film is about that perennial favourite of triumph against the odds.<br /><br />The acting is exemplary throughout. Stephen Dillane is excellent as usual, but the revelation of the movie is Shia LaBoeuf who delivers a disciplined, dignified and highly sympathetic performance as a working class Franco-Irish kid fighting his way through the prejudices of the New England WASP establishment. For those who are only familiar with his slap-stick performances in \"Even Stevens\" this demonstration of his maturity is a delightful surprise. And Josh Flitter as the ten year old caddy threatens to steal every scene in which he appears.<br /><br />A old fashioned movie in the best sense of the word: fine acting, clear directing and a great story that grips to the end - the final scene an affectionate nod to Casablanca is just one of the many pleasures that fill a great movie."
    ]


    # Evaluate the model on new texts
    evaluate_model(model, new_texts)
    
    # Upload model and get flag
    upload_model(model)


# Save the trained model to a file for future use
model_filename = 'capstone_model.joblib'
joblib.dump(model, model_filename)

print(f"Model saved to {model_filename}")
