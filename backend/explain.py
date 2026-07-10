"""Turns a trained TF-IDF + LogisticRegression pipeline's internals into a
plain-English explanation of a single prediction.

LogisticRegression gives us a coefficient per vocabulary term (positive =
pushes toward "phishing", negative = pushes toward "legit"). Multiplying
those coefficients by this document's TF-IDF values and sorting tells us
which words in *this specific email* did the most to produce its verdict -
a cheap, dependency-free stand-in for a full explainability library that's
accurate for linear models.
"""


def top_contributing_terms(pipeline, text, predicted_label, top_n=6):
    vectorizer = pipeline.named_steps["tfidf"]
    clf = pipeline.named_steps["clf"]

    X = vectorizer.transform([text])
    coefs = clf.coef_[0]
    contributions = X.multiply(coefs).tocsr()

    row = contributions.getrow(0)
    if row.nnz == 0:
        return []

    feature_names = vectorizer.get_feature_names_out()
    pairs = list(zip(row.indices, row.data))

    # For a "phishing" verdict we want the terms pushing hardest toward
    # phishing (positive contribution); for "legit" the terms pushing
    # hardest toward legit (most negative contribution).
    if predicted_label == 1:
        pairs.sort(key=lambda p: -p[1])
    else:
        pairs.sort(key=lambda p: p[1])

    terms = []
    for idx, value in pairs:
        if predicted_label == 1 and value <= 0:
            break
        if predicted_label == 0 and value >= 0:
            break
        terms.append(feature_names[idx])
        if len(terms) >= top_n:
            break
    return terms


def build_explanation(prediction, confidence, terms):
    pct = round(confidence * 100)
    if prediction == "fake":
        verdict_sentence = (
            f"PhishNet is {pct}% confident this email is phishing."
        )
        if terms:
            words = ", ".join(f"“{t}”" for t in terms)
            verdict_sentence += f" The strongest signals in the wording were: {words}."
        else:
            verdict_sentence += (
                " The overall writing style and structure matches known phishing "
                "emails, even without any single standout word."
            )
    else:
        verdict_sentence = (
            f"PhishNet is {pct}% confident this email is legitimate."
        )
        if terms:
            words = ", ".join(f"“{t}”" for t in terms)
            verdict_sentence += f" Its wording (e.g. {words}) reads like typical, low-risk correspondence."
        else:
            verdict_sentence += " No wording strongly associated with phishing was found."
    return verdict_sentence
