import os
from flask import Flask, flash, request, redirect, render_template, url_for, send_from_directory
from werkzeug.utils import secure_filename
import shutil
from extractor import Extractor


app=Flask(__name__)


app.secret_key = "secret key" # for encrypting the session
path = os.getcwd()

# file Upload
UPLOAD_FOLDER = os.path.join(path, 'uploads')
DOWNLOAD_FOLDER = os.path.join(path, 'download')

if not os.path.isdir(UPLOAD_FOLDER):
    os.mkdir(UPLOAD_FOLDER)

if not os.path.isdir(DOWNLOAD_FOLDER):
    os.mkdir(DOWNLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['DOWNLOAD_FOLDER'] = DOWNLOAD_FOLDER
ALLOWED_EXTENSIONS = set(['mp4'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def upload_form():
    return render_template('index.html')


@app.route('/', methods=['POST', 'GET'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No file selected for uploading')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            flash('File successfully processed')
            output_filename = process(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            shutil.move(output_filename, os.path.join(app.config['DOWNLOAD_FOLDER'], output_filename))
            return redirect(url_for('uploaded_file', output_filename = output_filename))
        else:
            flash('Allowed file types are video')
            return redirect(request.url)

def process(path):
    ext = Extractor()
    output_filename = ext.generate_highlights(path)
    return output_filename

@app.route('/view_highlights/<output_filename>')
def uploaded_file(output_filename):
    return render_template('download.html', output_filename = output_filename)
    #return send_from_directory(app.config['DOWNLOAD_FOLDER'], output_filename, as_attachment=True)

@app.route('/download_file/<output_filename>')
def download_file(output_filename):
    #output_filename = output_filename
    #print(output_filename)
    return send_from_directory(app.config['DOWNLOAD_FOLDER'], output_filename, as_attachment=True)

if __name__ == "__main__":
    app.run(host='0.0.0.0',port=5000, debug=True, use_reloader=False)
