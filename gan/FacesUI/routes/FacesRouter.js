const assert = require('assert');
const express = require('express');
const async = require('async');
const FacesLib = require("../lib/FacesLib.js")
const FS = require("../core/FS.js")
const Time = require("../core/Time.js")
const Validate = require("../core/Validate.js")
const router = express.Router();

// poij these guys should validate their inputs
router.get('/', GetFaces)
router.get('/similar', GetSimilarFaces)

/**
 * TODO. Make GetFaceByEncoding a POST, otw people can snoop the encoding in
 * transit (not secure).
 *
 * Get random faces or a single face by encoding depending on whether client
 * passes face encoding or not.
 * @param {*} req
 * @param {*} res
 * @param {*} next
 */
function GetFaces(req, res, next) {
    let { encoding } = req.body
    if (encoding) {
        encoding = Validate.sanitizeEncoding(encoding)
        GetFaceByEncoding(encoding, (er, img) => {
            if (er) next({ status: 500, error: "Cannot get face by encoding", er })
            else res.send({ status: 200, img })
        })
    } else {
        GetRandomFaces((er, img, encodings) => {
            if (er) next({ status: 500, error: "Cannot get random faces", er })
            else res.send({ status: 200, img, encodings })
        })
    }
}

/**
 *
 * @param {*} done(er, img, encodings)
 */
function GetRandomFaces(done) {
    const tag = "FacesRouter.GetRandomFaces"
    /** The images are in ../FacesFlask/out/GetRandomFaces-TIME/ */
    const pythonRootDir = "out/GetRandomFaces-" + Time.getTimeYYYYMMDDHHMMSSMS() // out/GetRandomFaces-2018-05-21-01-49-44-862-xDhYP
    const nodeRootDir = "../FacesFlask/" + pythonRootDir
    const imgFilename = "GetRandomFaces.jpg"
    const txtFilename = "GetRandomFaces.txt"
    const imgFilepath = nodeRootDir + "/" + imgFilename
    const txtFilepath = nodeRootDir + "/" + txtFilename
    let img, encodings
    async.waterfall([
        (done) => {
            FacesLib.GetRandomFaces(pythonRootDir, imgFilename, txtFilename, done)
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
        if (er) done({ tag, status: 500, error: "Cannot get random faces", er })
        else done(null, img, encodings)
        FS.rmdirf(nodeRootDir)
    })
}

/**
 * NOTE. Use this if you want fast random faces for testing.
 */
// function GetRandomFaces(done) {
//     const tag = "FacesRouter.GetRandomFaces"
//     const nodeRootDir = "out/GetRandomFacesMock"
//     const imgFilename = "GetRandomFaces.jpg"
//     const txtFilename = "GetRandomFaces.txt"
//     const imgFilepath = nodeRootDir + "/" + imgFilename
//     const txtFilepath = nodeRootDir + "/" + txtFilename
//     let img, encodings
//     async.waterfall([
//         (done) => {
//             FS.readImgFileAsBase64(imgFilepath, done)
//         }, (nimg, done) => {
//             img = nimg
//             FS.readTxtFile(txtFilepath, done)
//         }, (txt, done) => {
//             encodings = convertTextToEncodings(txt)
//             done()
//         }
//     ], (er) => {
//         if (er) done({ tag, status: 500, error: "Cannot get random faces", er })
//         else done(null, img, encodings)
//     })
// }

function convertTextToEncodings(txt) {
    assert(typeof txt == "string", "Text encoding should be a string")
    assert(txt.length > 0, "Text encoding should not have zero length")
    const encodings = txt.split("\n").map(line => {
        const encoding = line.split(" ")
            .map(num => parseFloat(num))
            .filter(num => !isNaN(num)) // Filter out empty last line
        return encoding
    }).filter(encoding => encoding.length > 0) // Filter out empty last line
    return encodings
}

/**
 * Get a face by its encoding.
 * @param {*} encoding
 * @param {*} done(er, img)
 */
function GetFaceByEncoding(encoding, done) {
    // qwer. Check that entries in the encoding are floats and in the right
    // range [-1, 1].
    /** The images are in ../FacesFlask/out/GetFaceByEncoding-TIME/ */
    const pythonRootDir = "out/GetFaceByEncoding-" + Time.getTimeYYYYMMDDHHMMSSMS() // out/GetFaceByEncoding-2018-05-21-01-49-44-862-xDhYP
    const nodeRootDir = "../FacesFlask/" + pythonRootDir
    const imgFilename = "GetFaceByEncoding.jpg"
    const txtFilename = "GetFaceByEncoding.txt"
    const imgFilepath = nodeRootDir + "/" + imgFilename
    async.waterfall([
        (done) => {
            FacesLib.GetFaceByEncoding(encoding, pythonRootDir, imgFilename, txtFilename, done)
        }, (done) => {
            FS.readImgFileAsBase64(imgFilepath, done)
        }, (img, done) => {
            done(null, img)
        }
    ], (er, img) => {
        if (er) done(er)
        else done(null, img)
        FS.rmdirf(nodeRootDir)
    })
}

/**
 * TODO. Validate inputs, e.g. encoding
 * @param {*} req
 * @param {*} res
 * @param {*} next
 */
function GetSimilarFaces(req, res, next) {
    const tag = "FacesRouter.GetSimilarFaces"
    /** The images are in ../FacesFlask/out/GetSimilarFaces-TIME/ */
    const pythonRootDir = "out/GetSimilarFaces-" + Time.getTimeYYYYMMDDHHMMSSMS() // out/GetSimilarFaces-2018-05-21-01-49-44-862-xDhYP
    const nodeRootDir = "../FacesFlask/" + pythonRootDir
    const imgFilename = "GetSimilarFaces.jpg"
    const txtFilename = "GetSimilarFaces.txt"
    const imgFilepath = nodeRootDir + "/" + imgFilename
    const txtFilepath = nodeRootDir + "/" + txtFilename
    const encoding = Validate.sanitizeEncoding(req.body.encoding)
    let img, encodings
    async.waterfall([
        (done) => {
            FacesLib.GetSimilarFaces(encoding, pythonRootDir, imgFilename, txtFilename, done)
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
        if (er) next({ tag, status: 500, error: "Cannot get similar faces", er })
        else res.send({ tag, status: 200, img, encodings })
        FS.rmdirf(nodeRootDir)
    })
}

module.exports = router;