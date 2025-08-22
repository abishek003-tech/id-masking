import os
from datetime import datetime
from flask import Flask, request, render_template, send_from_directory, url_for, redirect, flash
from werkzeug.utils import secure_filename

from utils.ocr_mask import mask_sensitive_info

# ------------------ Config ------------------
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
BASE_DIR = os.path.dirname(os.path.abspath(__file__))   # ✅ fixed __file__
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
MASKED_DIR = os.path.join(BASE_DIR, "masked")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(MASKED_DIR, exist_ok=True)

app = Flask(__name__)   # ✅ fixed __name__
app.config["UPLOAD_FOLDER"] = UPLOAD_DIR
app.config["MASKED_FOLDER"] = MASKED_DIR
app.secret_key = "change-this-in-production"  # for flash messages

# ------------------ Helpers ------------------
def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

# ------------------ Routes ------------------
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            flash("No file part in the request.")
            return redirect(request.url)

        file = request.files["file"]
        if not file or file.filename == "":
            flash("No file selected.")
            return redirect(request.url)

        if not allowed_file(file.filename):
            flash("Only .jpg, .jpeg, .png are allowed.")
            return redirect(request.url)

        # save upload
        safe_name = secure_filename(file.filename)
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        filename = f"{ts}-{safe_name}"
        input_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(input_path)

        # process/mask
        output_path, counts = mask_sensitive_info(input_path, os.path.join(app.config["MASKED_FOLDER"], filename))

        masked_url = url_for("masked_file", filename=os.path.basename(output_path))
        return render_template("index.html",
                               masked_url=masked_url,
                               counts=counts,
                               filename=os.path.basename(output_path))

    return render_template("index.html")

@app.route("/masked/<path:filename>")
def masked_file(filename):
    return send_from_directory(app.config["MASKED_FOLDER"], filename, as_attachment=False)

@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename, as_attachment=False)

# ------------------ Entrypoint ------------------
if __name__ == "__main__":    # ✅ fixed __main__
    # Run:  python app.py   (Windows)  or  python3 app.py
    app.run(debug=True)
