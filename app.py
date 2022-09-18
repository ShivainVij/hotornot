from flask import Flask, jsonify
from flask_restful import Resource, Api, reqparse
from werkzeug.datastructures import FileStorage
from predict import predict
import tempfile

app = Flask(__name__)
app.logger.setLevel('INFO')

api = Api(app)

parser = reqparse.RequestParser()
parser.add_argument('file',
                    type=FileStorage,
                    location='files',
                    required=True,
                    help='provide a file')

class Image(Resource):

    def post(self):
        args = parser.parse_args()
        the_file = args['file']
        # save a temporary copy of the file
        ofile, ofname = tempfile.mkstemp()
        the_file.save(ofname)
        # predict
        results = predict(ofname)
        # formatting the results as a JSON-serializable structure:

        return jsonify(results)


api.add_resource(Image, '/image')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)