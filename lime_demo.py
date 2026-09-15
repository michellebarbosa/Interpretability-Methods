"""
LIME for text classification — atheism vs. christian newsgroups.
Adapted from the official LIME text tutorial.
"""

import numpy as np
import sklearn
import sklearn.ensemble
import sklearn.metrics
from sklearn.datasets import fetch_20newsgroups
from sklearn.pipeline import make_pipeline
from lime.lime_text import LimeTextExplainer

# 1. Fetch data — 2-class subset
categories = ['alt.atheism', 'soc.religion.christian']
newsgroups_train = fetch_20newsgroups(subset='train', categories=categories)
newsgroups_test = fetch_20newsgroups(subset='test', categories=categories)
class_names = ['atheism', 'christian']

# 2. Vectorize text
vectorizer = sklearn.feature_extraction.text.TfidfVectorizer(lowercase=False)
train_vectors = vectorizer.fit_transform(newsgroups_train.data)
test_vectors = vectorizer.transform(newsgroups_test.data)

# 3. Train a random forest (hard to interpret directly)
rf = sklearn.ensemble.RandomForestClassifier(n_estimators=500)
rf.fit(train_vectors, newsgroups_train.target)

pred = rf.predict(test_vectors)
f1 = sklearn.metrics.f1_score(newsgroups_test.target, pred, average='binary')
print(f"F1 score: {f1:.3f}")

# 4. Wrap vectorizer + model in a pipeline so LIME can work on raw text
c = make_pipeline(vectorizer, rf)
print(c.predict_proba([newsgroups_test.data[0]]))

# 5. Explain one prediction
explainer = LimeTextExplainer(class_names=class_names)

idx = 55
exp = explainer.explain_instance(
    newsgroups_test.data[idx], c.predict_proba, num_features=6
)

print(f"Document id: {idx}")
print("Probability(christian) =", c.predict_proba([newsgroups_test.data[idx]])[0, 1])
print(f"True class: {class_names[newsgroups_test.target[idx]]}")

print("\nExplanation (feature, weight):")
for feature, weight in exp.as_list():
    print(f"  {feature:15s} {weight:+.4f}")

# 6. Sanity check: manually remove top features, see prediction shift
original_pred = rf.predict_proba(test_vectors[idx])[0, 1]
tmp = test_vectors[idx].copy().tolil()
for word in ['Posting', 'Host']:
    if word in vectorizer.vocabulary_:
        tmp[0, vectorizer.vocabulary_[word]] = 0
tmp = tmp.tocsr()

new_pred = rf.predict_proba(tmp)[0, 1]
print(f"\nOriginal prediction: {original_pred:.3f}")
print(f"Prediction removing top features: {new_pred:.3f}")
print(f"Difference: {new_pred - original_pred:.3f}")

# 7. Save visual explanation
exp.save_to_file("/Users/michelle/Desktop/Interpretability-Methods/lime_text_explanation.html")
print("\nSaved visual explanation to lime_text_explanation.html")