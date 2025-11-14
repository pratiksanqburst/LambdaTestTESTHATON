# simple flask app used as a sample source for route extraction
from flask import Flask, jsonify, request

app = Flask(__name__)

@app.route('/items', methods=['GET'])
def list_items():
    return jsonify([]), 200

@app.route('/items', methods=['POST'])
def create_item():
    data = request.get_json()
    if not data or 'name' not in data:
        return jsonify({'error':'bad'}), 400
    return jsonify({'id':1, 'name': data['name']}), 201

@app.route('/items/<int:item_id>', methods=['GET'])
def get_item(item_id):
    return jsonify({'id': item_id, 'name': 'x'}), 200
@app.route('/items/<int:item_id>', methods=['GET'])
def get_item(item_id):
    return jsonify({'id': item_id, 'name': 'x'}), 200


@app.route('/items/<int:item_id>', methods=['GET'])
def get_item(item_id):
    return jsonify({'id': item_id, 'name': 'x'}), 200

if __name__ == '__main__':
    app.run(port=5000)
    
