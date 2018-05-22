const assert = require('assert');
const express = require('express');
const async = require('async');
const Faces = require("../lib/Faces.js")
const FS = require("../core/FS.js")
const Time = require("../core/Time.js")
const router = express.Router();

router.get('/', getRandomFaces)

function getRandomFaces(req, res, next) {
    const tag = "Faces.getRandomFaces"
    const nodeRootDir = "out/getRandomFaces-" + Time.getTimeYYYYMMDDHHMMSSMS() // out/getRandomFaces-2018-05-21-01-49-44-862-xDhYP
    const pythonRootDir = "FacesUI/" + nodeRootDir // python cwd is ai/gan/FacesUI/, one above node's root
    const imgFilename = "getRandomFaces.jpg"
    const txtFilename = "getRandomFaces.txt"
    const imgFilepath = nodeRootDir + "/" + imgFilename
    const txtFilepath = nodeRootDir + "/" + txtFilename
    let img, encodings
    async.waterfall([
        (done) => {
            Faces.makeRandomFaces(pythonRootDir, imgFilename, txtFilename, done)
        }, (done) => {
            FS.readImgFileAsBase64(imgFilepath, done)
        }, (nimg, done) => {
            img = nimg
            FS.readTxtFile(txtFilepath, done)
        }, (txt, done) => {
            encodings = convertTextToEncodings(txt)
            done()
        }
    ], (er) => {
        if (er) next({ status: 500, error: "Cannot create random faces", er })
        else res.send({ status: 200, img, encodings })
        FS.rmdirf(dir)
    })
}

function convertTextToEncodings(txt) {
    assert(typeof txt == "string", "Text encoding should be a string")
    assert(txt.length > 0, "Text encoding should not have zero length")
    const encodings = txt.split("\n").map(line => {
        const encoding = line.split(" ").map(num => parseFloat(num))
            .filter(num => !isNaN(num)) // Filter out empty last line
        return encoding
    }).filter(encoding => encoding.length > 0) // Filter out empty last line
    return encodings
}

module.exports = router;
