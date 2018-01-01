
from flask import Flask, request

app = Flask(__name__)

@app.route('/api', methods=['POST'])
def hello_world_api():
    data = request.get_json(force=True)
    name = data['name']
    return "Hello to API World : {0}".format(name)

if __name__ == "__main__":
    app.run(port=8292, debug=True)