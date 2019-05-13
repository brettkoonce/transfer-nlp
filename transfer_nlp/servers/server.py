from flask import jsonify, send_from_directory, request, Response, Flask
import logging
import os

logging.getLogger('gensim').setLevel(logging.ERROR)

app = Flask(__name__)
app.root_path = os.path.dirname(os.path.abspath(__file__))


@app.route("/demo")
def priority():
    return send_from_directory('web', 'demo.html')


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3000)
