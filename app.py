from flask import Flask, jsonify, request
import cbText

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False

@app.route('/')
def hello_world():  # put application's code here
    return 'Hello World!'


@app.route('/category/<string:name>', methods=['GET'])
def test(name):
    result = cbText.cb_fast_text(name)
    return result


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)
