from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__, template_folder='templates')

# Load model and mappings
with open("model_weights/svd_model/model.pkl", "rb") as f:
    model = pickle.load(f)

with open("model_weights/svd_model/inner2title.pkl", "rb") as f:
    inner2title = pickle.load(f)
    title2inner = {t: i for i, t in inner2title.items()}

@app.route("/", methods=["GET", "POST"])
def recommend():
    recommendations = []
    query = ""
    if request.method == "POST":
        query = request.form["book_title"]
        recommendations = get_recommendations(query)
    return render_template("home.html", recommendations=recommendations, query=query)

def get_recommendations(book_title, top_n=10):
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
        (inner2title[iid], round(score, 3))
        for iid, score in enumerate(sims)
        if iid != inner_id
    ]
    recs.sort(key=lambda x: x[1], reverse=True)
    return recs[:top_n]

if __name__ == "__main__":
    app.run(debug=True)
