"""
Compare LIME explanations across different ML techniques on the same text
classification task (atheism vs. christian newsgroups).

Goal: see whether different model types rely on the same shortcut features
(email headers) or pick up genuinely different signals.
"""

import sklearn
import sklearn.feature_extraction.text
import sklearn.metrics
from sklearn.datasets import fetch_20newsgroups
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from lime.lime_text import LimeTextExplainer

# 1. Load data
categories = ['alt.atheism', 'soc.religion.christian']
newsgroups_train = fetch_20newsgroups(subset='train', categories=categories)
newsgroups_test = fetch_20newsgroups(subset='test', categories=categories)
class_names = ['atheism', 'christian']

vectorizer = sklearn.feature_extraction.text.TfidfVectorizer(lowercase=False)
train_vectors = vectorizer.fit_transform(newsgroups_train.data)
test_vectors = vectorizer.transform(newsgroups_test.data)

# 2. Define several different model types
# LinearSVC has no predict_proba, so wrap it with a calibrator
models = {
    "Random Forest": RandomForestClassifier(n_estimators=500, random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Naive Bayes": MultinomialNB(),
    "Linear SVM (calibrated)": CalibratedClassifierCV(LinearSVC(), cv=3),
}

trained = {}
for name, model in models.items():
    model.fit(train_vectors, newsgroups_train.target)
    pred = model.predict(test_vectors)
    f1 = sklearn.metrics.f1_score(newsgroups_test.target, pred, average='binary')
    print(f"{name:28s} F1 = {f1:.3f}")
    trained[name] = model

# 3. Explain the SAME document across all models
idx = 83
document = newsgroups_test.data[idx]
true_class = class_names[newsgroups_test.target[idx]]
print(f"\nDocument id: {idx} | True class: {true_class}\n")

explainer = LimeTextExplainer(class_names=class_names)

for name, model in trained.items():
    pipeline = make_pipeline(vectorizer, model)
    prob_christian = pipeline.predict_proba([document])[0, 1]

    exp = explainer.explain_instance(document, pipeline.predict_proba, num_features=6)

    print(f"--- {name} ---")
    print(f"Probability(christian) = {prob_christian:.3f}")
    for feature, weight in exp.as_list():
        print(f"  {feature:15s} {weight:+.4f}")
    print()

    exp.save_to_file(f"/Users/michelle/Desktop/Interpretability-Methods/{name.replace(' ', '_').lower()}.html")

print("Saved one HTML explanation per model to outputs.")