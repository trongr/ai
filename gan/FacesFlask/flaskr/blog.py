import sys
sys.path.append('../')
sys.path.append('../../')
sys.path.append("../../utils/")
import Faces
from flask import Blueprint, flash, g, redirect, render_template, request, url_for, jsonify
from werkzeug.exceptions import abort

bp = Blueprint('blog', __name__)


@bp.route('/')
def index():
    return render_template('blog/index.html')


# poij reduce dimension of random faces grid and see if that makes generation
# faster.
@bp.route('/RandomFaces', methods=['GET'])
def GetRandomFaces():
    outputDir = "out"
    imgFilename = "RandomFaces.jpg"
    txtFilename = "RandomFaces.txt"
    Faces.MakeRandomFaces(outputDir, imgFilename, txtFilename)
    return jsonify({"status": 200})
