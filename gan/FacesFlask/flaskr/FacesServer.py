import sys
sys.path.append('../')
sys.path.append('../../')
sys.path.append("../../utils/")
import Faces
from flask import Blueprint, flash, g, redirect, render_template, request, url_for, jsonify
from werkzeug.exceptions import abort

bp = Blueprint('FacesServer', __name__)


@bp.route('/')
def index():
    return render_template('FacesServer/index.html')


@bp.route('/GetRandomFaces', methods=['GET'])
def GetRandomFaces():
    outputDir = request.args.get("outputDir")
    imgFilename = request.args.get("imgFilename")
    txtFilename = request.args.get("txtFilename")
    Faces.MakeRandomFaces(outputDir, imgFilename, txtFilename)
    return jsonify({"status": 200})


@bp.route('/GetSimilarFaces', methods=['GET'])
def GetSimilarFaces():
    args = request.get_json()
    encoding = args["encoding"]
    outputDir = args["outputDir"]
    imgFilename = args["imgFilename"]
    txtFilename = args["txtFilename"]
    Faces.MakeSimilarFaces(encoding, outputDir, imgFilename, txtFilename)
    return jsonify({"status": 200})


@bp.route('/GetFaceByEncoding', methods=['GET'])
def GetFaceByEncoding():
    args = request.get_json()
    encoding = args["encoding"]
    outputDir = args["outputDir"]
    imgFilename = args["imgFilename"]
    txtFilename = args["txtFilename"]
    Faces.MakeFaceByEncoding(encoding, outputDir, imgFilename, txtFilename)
    return jsonify({"status": 200})
