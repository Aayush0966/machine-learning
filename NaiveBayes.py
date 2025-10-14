import math
from collections import defaultdict
import random

data = [
    ("buy cheap meds now", "spam"),
    ("cheap meds available today", "spam"),
    ("limited time offer just for you", "spam"),
    ("win money now", "spam"),
    ("click here to claim your prize", "spam"),
    ("cheap loans available instantly", "spam"),
    ("free trial just for you", "spam"),
    ("win a free vacation now", "spam"),
    ("offer ends today", "spam"),
    ("click here to subscribe", "spam"),
    ("congratulations you won a gift card", "spam"),
    ("urgent: claim your reward", "spam"),
    ("exclusive deal for you", "spam"),
    ("earn cash quickly", "spam"),
    ("act now to get free coupons", "spam"),
    ("meeting schedule tomorrow", "not_spam"),
    ("project deadline extended", "not_spam"),
    ("lunch with team today", "not_spam"),
    ("update on the report", "not_spam"),
    ("team outing next week", "not_spam"),
    ("your invoice is attached", "not_spam"),
    ("urgent: please review the document", "not_spam"),
    ("schedule call with client", "not_spam"),
    ("monthly performance report", "not_spam"),
    ("don’t forget the meeting", "not_spam"),
    ("team meeting rescheduled", "not_spam"),
    ("send the presentation slides", "not_spam"),
    ("please approve the leave request", "not_spam"),
    ("follow up on the client email", "not_spam"),
    ("review the quarterly report", "not_spam"),
]

def tokenize(text):
    return text.lower().split()


vocab = set()

for text, label in data:
    vocab.update(tokenize(text))

words_count = defaultdict(int)
classes_count = defaultdict(lambda: defaultdict(int))

for text, label in data:
    words_count[label] += 1
    for word in tokenize(text):
        classes_count[label][word] += 1
        




def predict(text):
    words = tokenize(text)
    docs = defaultdict(int)
    
    for label in classes_count.keys():
        docs[label] += sum(classes_count[label].values())
    
    
    priors = {label: docs[label] / sum(docs.values()) for label in words_count}
    
    class_probs = {}
    
    for label in classes_count.keys():
        
        class_probs[label] = math.log(priors[label])
        
        for word in words:
            word_count = classes_count[label][word]
            likelihood = (word_count + 1) / (sum(classes_count[label].values()) + len(vocab))
            class_probs[label] += math.log(likelihood)
            
    
    return max(class_probs, key=class_probs.get), class_probs
            
    
    
test_text = "please review the free project report"
predicted_label, class_probs = predict(test_text)

print(f"Text: '{test_text}'")
print(f"Predicted Label: {predicted_label}")
print("Class Probabilities:", class_probs)