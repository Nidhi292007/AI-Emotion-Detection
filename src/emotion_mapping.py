# src/emotion_mapping.py

emotion_labels = [
    "admiration","amusement","anger","annoyance","approval","caring",
    "confusion","curiosity","desire","disappointment","disapproval",
    "disgust","embarrassment","excitement","fear","gratitude","grief",
    "joy","love","nervousness","optimism","pride","realization",
    "relief","remorse","sadness","surprise","neutral"
]

emotion_map = {
    "admiration":"happy",
    "amusement":"happy",
    "approval":"happy",
    "caring":"happy",
    "desire":"happy",
    "excitement":"happy",
    "gratitude":"happy",
    "joy":"happy",
    "love":"happy",
    "optimism":"happy",
    "pride":"happy",
    "relief":"happy",

    "anger":"angry",
    "annoyance":"angry",
    "disapproval":"angry",

    "fear":"fear",
    "nervousness":"fear",

    "disappointment":"sad",
    "grief":"sad",
    "remorse":"sad",
    "sadness":"sad",

    "confusion":"surprise",
    "curiosity":"surprise",
    "realization":"surprise",
    "surprise":"surprise",

    "disgust":"disgust",
    "embarrassment":"disgust",

    "neutral":"neutral"
}

final_labels = {
    "happy":0,
    "sad":1,
    "angry":2,
    "fear":3,
    "surprise":4,
    "disgust":5,
    "neutral":6
}