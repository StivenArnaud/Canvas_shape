import cv2
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from canvas import Canvas, CANVAS_SIZES, border_bool

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def canvas_filling():
    if request.method == 'POST':
        uploaded_file = request.files.get('file')

        if not uploaded_file:
            return render_template('index.html', error="No file uploaded")
        action_selected = None
        if 'placer' in request.form:
            action_selected = 'placer'
        elif 'border' in request.form:
            action_selected = 'border'
        elif 'decouper' in request.form:
            action_selected = 'decouper'
        format_selected = request.form.get('format')

        # Create a Canvas object with the selected format size
        print(f"here {format_selected}")
        canvas_width, canvas_height = CANVAS_SIZES[format_selected.upper()]
        canvas_obj = Canvas(canvas_width, canvas_height)

        # Set the border option based on user's choice
        canvas_obj.set_border_option(action_selected)
        # Process the uploaded file
        uploaded_file.seek(0)
        uploaded_image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_UNCHANGED)
        uploaded_file.seek(0)
        # Load the uploaded image as a shape
        shapes = canvas_obj.load_shapes_modified([uploaded_file]) # Ensure this method exists
        print(f"shapes {shapes}")
        # Call the main_new_approach function to process and generate the canvas
        uploaded_file.seek(0)
        output_path = canvas_obj.main_new_approach([uploaded_file])
        if output_path:
            return redirect(url_for('download_file', filename=output_path.split('/')[-1]))
        else:
            return render_template('index.html', error="An error occurred while processing the image")

    return render_template('index.html')

@app.route('/download/<filename>')
def download_file(filename):
    response = send_from_directory('.', filename, as_attachment=True)
    # Add headers to signal the browser to redirect after download
    response.headers["Content-Disposition"] = f"attachment; filename={filename};"
    response.headers["Refresh"] = "5; url=/"  # Redirect to the main page after 5 seconds
    return response

if __name__ == "__main__":
    app.run(debug=True)
