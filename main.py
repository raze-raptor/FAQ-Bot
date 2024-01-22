import spacy
from spacy import displacy
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json

# Load English tokenizer, tagger, parser, NER and word vectors
nlp = spacy.load("en_core_web_sm")

# Load data
with open('faq.json') as f:
    data = json.load(f)

# Get all questions and answers
questions = [item['question'] for item in data]
answers = [item['answer'] for item in data]

# Create a vectorizer
vectorizer = TfidfVectorizer()

# Convert questions to vectors
X = vectorizer.fit_transform(questions)

def get_answer(question):
    # Convert question to vector
    question_vec = vectorizer.transform([question])
    
    # Compute similarity scores
    similarity_scores = cosine_similarity(question_vec, X).flatten()
    
    # Get index of most similar question
    most_similar_index = similarity_scores.argsort()[::-1][0]
    
    # Return corresponding answer
    return answers[most_similar_index]

# Test the function
print(get_answer("Your question here"))
