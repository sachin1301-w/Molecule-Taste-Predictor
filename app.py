from flask import Flask, render_template, request
import joblib

# Load model and vectorizer
model = joblib.load("sb_model.pkl")
vectorizer = joblib.load("sb_vectorizer.pkl")

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    smiles_input = ""
    
    if request.method == "POST":
        smiles_input = request.form["smiles"]
        
        if smiles_input.strip():
            # Transform SMILES using vectorizer
            X_vec = vectorizer.transform([smiles_input])
            # Predict taste
            pred = model.predict(X_vec)[0]
            prediction = f"Predicted Taste: {pred}"
        else:
            prediction = "⚠️ Please enter a SMILES string."

    return render_template("index.html", prediction=prediction, smiles_input=smiles_input)

if __name__ == "__main__":
    app.run(debug=True)
