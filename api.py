from flask import Flask, render_template, request
from thefuzz import process

import pickle
import torch
import numpy as np
import pandas as pd

app = Flask(__name__, template_folder='templates')

# # Load model and mappings
# with open("model_weights/svd/model.pkl", "rb") as f:
#     svd_model = pickle.load(f)
#
# with open("model_weights/svd/inner2title.pkl", "rb") as f:
#     inner2title = pickle.load(f)
#     title2inner = {t: i for i, t in inner2title.items()}

with open("model_weights/ease/model_ease_imp_b10u3r100.pkl", "rb") as f:
    ease_model = pickle.load(f)

with open("model_weights/ease/df_ease_imp_b10u3r100.pkl", "rb") as f:
    ease_dataframe = pickle.load(f)

@app.route("/", methods=["GET", "POST"])
def recommend():
    recommendations = []
    query = ''
    if request.method == 'POST':
        query = request.form['book_title']
        model = request.form['model']
        if model.lower() == 'svd':
            print('ok')
            # recommendations = get_recommendations_svd(query, svd_model, title2inner, inner2title)
        elif model.lower() == 'ease':
            recommendations = get_recommendations_ease(query, ease_model, ease_dataframe)

    return render_template("home.html", recommendations=recommendations, query=query)

def get_recommendations_svd(book_title, model, title2inner, inner2title, top_n=10):
    key = book_title.lower()
    if key not in title2inner:
        for inner_id, title in inner2title.items():
            if key in title:
                key = title
                break
        else:
            return [("No match found", 0)]

    inner_id = title2inner[key]
    q = model.qi[inner_id]
    q_norm = q / np.linalg.norm(q)
    db_norm = model.qi / np.linalg.norm(model.qi, axis=1, keepdims=True)
    sims = db_norm.dot(q_norm)

    recs = [
        inner2title[iid]
        for iid, _ in enumerate(sims)
        if iid != inner_id
    ]
    recs.sort(key=lambda x: x[1], reverse=True)
    return recs[:top_n]

@app.route("/", methods=["GET", "POST"])
def get_recommendations_ease(book_title, B, df, score_cutoff=90, top_n=10):
    unique_titles = list(dict.fromkeys(df['Book-Title'].str.lower()))
    unique_isbns = df['ISBN'].unique().tolist()
    
    best_match = process.extractOne(book_title.lower(), unique_titles)
    if not best_match or best_match[1] < score_cutoff:
        return None

    print(f'Matched book: {best_match[0]}')
    matched_ISBNs = df[df['Book-Title'].str.lower() == best_match[0]]['ISBN']
    if matched_ISBNs.empty:
        return None

    inner_id = unique_isbns.index(matched_ISBNs.iloc[0])
    top_indices = torch.topk(B[inner_id], top_n).indices

    top_indices = top_indices.tolist() 
    top_isbns = [unique_isbns[i] for i in top_indices]
    titles = df[df['ISBN'].isin(top_isbns)]['Book-Title'].unique().tolist()
    return titles

if __name__ == "__main__":
    app.run(debug=True)
