from flask import Flask, render_template, request
from clustering import proses_clustering
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/proses', methods=['POST'])
def proses():
    file = request.files['filecsv']
    if file and file.filename != '':
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        df = proses_clustering(filepath)
        if df is not None:
            table_html = df.to_html(classes='table table-bordered', index=False)
            return render_template(
                'index.html',
                tabel=table_html,
                gambar='static/hasil.png',
                elbow='static/elbow.png',
                silhouette='static/silhouette.png'
            )
        else:
            return render_template('index.html', tabel="Gagal memproses file.")
    return render_template('index.html', tabel="Tidak ada file yang diunggah.")
from flask import send_file

@app.route('/download/csv')
def download_csv():
    return send_file("static/hasil_clustering.csv", as_attachment=True)

@app.route('/download/gambar/<jenis>')
def download_gambar(jenis):
    if jenis == "hasil":
        path = "static/hasil.png"
    elif jenis == "elbow":
        path = "static/elbow.png"
    elif jenis == "silhouette":
        path = "static/silhouette.png"
    else:
        return "Gambar tidak ditemukan", 404

    return send_file(path, as_attachment=True)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
