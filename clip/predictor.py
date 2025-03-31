# This is the file that implements a flask server to do inferences. It's the file that you will modify to
# implement the scoring for your own algorithm.

from __future__ import print_function

import os
from flask import Flask, jsonify, request, Response
import json

from sentence_transformers import SentenceTransformer

prefix = "/opt/ml/"
model_path = os.path.join(prefix, "model")

# A singleton for holding the model. This simply loads the model and holds it.
# It has a predict function that does a prediction based on the model and the input data.


class clip_embedding(object):
    model = None  # Where we keep the model when it's loaded

    @classmethod
    def get_model(cls):
        """Get the model object for this instance, loading it if it's not already loaded."""
        if cls.model == None:
            cls.model = SentenceTransformer(model_path)
        return cls.model

    @classmethod
    def encode(cls, input):
        clf = cls.get_model()
        return clf.encode(input)


# The flask app for serving predictions
app = Flask(__name__)


@app.route("/ping", methods=["GET"])
def ping():
    """Determine if the container is working and healthy. In this sample container, we declare
    it healthy if we can load the model successfully."""
    health = clip_embedding.get_model() is not None  # You can insert a health check here

    status = 200 if health else 404
    return "Healthy", status


@app.route("/invocations", methods=["POST"])
def transformation():
    data = None

    # Convert from CSV to pandas
    if request.content_type == "application/json":
        data = json.loads(request.data.decode("utf-8"))
    else:
        return Response(
            response="This predictor only supports JSON data", status=415, mimetype="text/plain"
        )
    embeddings = clip_embedding.encode(data["search_text"])
    result = {
        "embeddings":embeddings.tolist()
    }
    return jsonify(result)
