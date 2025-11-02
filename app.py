from flask import Flask, request, send_file
import os
import cv2

# Import individual style modules from the 'styles' folder
from style_filters.cartoon import cartoonize_image
from style_filters.sketch import sketch_image
from style_filters.watercolor import watercolor_image
#from style_filters.pixel import pixelate_image

app = Flask(__name__, static_folder='static')

# Create upload/output directories
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

@app.route('/')
def home():
    return app.send_static_file('index.html')

@app.route('/api/convert', methods=['POST'])
def convert():
    try:
        # Get uploaded file and selected style
        file = request.files['image']
        style = request.form.get('style', 'cartoon')

        input_path = os.path.join(UPLOAD_FOLDER, file.filename)
        output_path = os.path.join(OUTPUT_FOLDER, "converted_" + file.filename)
        file.save(input_path)

        # Route to the correct transformation function
        if style == 'cartoon':
            cartoonize_image(input_path, output_path)
        elif style == 'sketch':
            sketch_image(input_path, output_path)
        elif style == 'watercolor':
            watercolor_image(input_path, output_path)
        elif style == 'pixel':
            pixelate_image(input_path, output_path)
        else:
            cartoonize_image(input_path, output_path)

        # Return processed image
        return send_file(output_path, mimetype='image/png')
    except Exception as e:
        return {"error": str(e)}, 500


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
