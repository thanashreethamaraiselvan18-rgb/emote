import re
from sklearn.metrics import accuracy_score, classification_report
emotion_lexicon = {
    # Happy
    "happy": "happy", "joy": "happy", "joyful": "happy", "excited": "happy",
    "great": "happy", "pleased": "happy", "smile": "happy", "delighted": "happy",

    # Sad
    "sad": "sad", "depressed": "sad", "cry": "sad", "crying": "sad",
    "lonely": "sad", "unhappy": "sad", "down": "sad", "hopeless": "sad",

    # Anger
    "angry": "anger", "anger": "anger", "mad": "anger",
    "furious": "anger", "hate": "anger", "annoyed": "anger", "irritated": "anger",

    # Fear
    "fear": "fear", "scared": "fear", "afraid": "fear",
    "nervous": "fear", "terrified": "fear", "frightened": "fear", "panic": "fear",

    # Surprise
    "surprise": "surprise", "shocked": "surprise", "amazed": "surprise",
    "astonished": "surprise", "unexpected": "surprise", "wow": "surprise",

    # Love
    "love": "love", "loved": "love", "loving": "love",
    "care": "love", "affection": "love", "romantic": "love", "dear": "love",

    # Disgust
    "disgust": "disgust", "disgusted": "disgust", "gross": "disgust",
    "dirty": "disgust", "nasty": "disgust", "awful": "disgust"
}
texts = [
    # Happy
    "I am very happy today",
    "This makes me feel joyful",
    "I am delighted with the result",

    # Sad
    "I feel sad and lonely",
    "This is a hopeless situation",
    "I am crying today",

    # Anger
    "I am very angry right now",
    "I hate this situation",
    "This makes me furious",

    # Fear
    "I am scared of the exam",
    "I feel nervous",
    "This is frightening",

    # Surprise
    "Wow this is unexpected",
    "I am shocked by the news",
    "That was an amazing surprise",

    # Love
    "I love my family",
    "I care deeply about you",
    "This is a romantic moment",

    # Disgust
    "This food is disgusting",
    "That smell is awful",
    "The place looks dirty"
]

true_emotions = [
    "happy","happy","happy",
    "sad","sad","sad",
    "anger","anger","anger",
    "fear","fear","fear",
    "surprise","surprise","surprise",
    "love","love","love",
    "disgust","disgust","disgust"
]

# --------------------------------------------
# TEXT PREPROCESSING
# --------------------------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z ]", "", text)
    return text.split()

# --------------------------------------------
# EMOTION PREDICTION FUNCTION
# --------------------------------------------
def predict_emotion(text):
    words = clean_text(text)
    detected = []

    for word in words:
        if word in emotion_lexicon:
            detected.append(emotion_lexicon[word])

    if detected:
        return max(set(detected), key=detected.count)
    else:
        return "neutral"

# --------------------------------------------
# DATASET PREDICTION (FOR METRICS)
# --------------------------------------------
predicted_emotions = [predict_emotion(sentence) for sentence in texts]

# --------------------------------------------
# USER INPUT
# --------------------------------------------
user_input = input("Enter a sentence to detect emotion: ")
user_prediction = predict_emotion(user_input)

print("\n🔍 Emotion Detection Result")
print("----------------------------")
print("User Input        :", user_input)
print("Predicted Emotion :", user_prediction)

# --------------------------------------------
# ACCURACY & CLASSIFICATION REPORT (AFTER INPUT)
# --------------------------------------------
accuracy = accuracy_score(true_emotions, predicted_emotions)

print("\n📊 Model Evaluation Metrics")
print("----------------------------")
print("Accuracy:", format(accuracy, ".2f"))
print("Classification Report:\n")
print(classification_report(
    true_emotions,
    predicted_emotions,
    zero_division=0
))
