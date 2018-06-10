const assert = require('assert');
const express = require('express');
const async = require('async');
const Faces = require("../lib/Faces.js")
const FS = require("../core/FS.js")
const Time = require("../core/Time.js")
const Validate = require("../core/Validate.js")
const router = express.Router();

router.get('/', getFaces)
router.get('/similar', getSimilarFaces)

/**
 * TODO. Make getFaceByEncoding a POST, otw people can snoop the encoding in
 * transit (not secure).
 *
 * Get random faces or a single face by encoding depending on whether client
 * passes face encoding or not.
 * @param {*} req
 * @param {*} res
 * @param {*} next
 */
function getFaces(req, res, next) {
    let { encoding } = req.body
    if (encoding) {
        encoding = Validate.sanitizeEncoding(encoding)
        getFaceByEncoding(encoding, (er, img) => {
            if (er) next({ status: 500, error: "Cannot get face by encoding", er })
            else res.send({ status: 200, img })
        })
    } else {
        getRandomFaces((er, img, encodings) => {
            if (er) next({ status: 500, error: "Cannot get random faces", er })
            else res.send({ status: 200, img, encodings })
        })
    }
}

// poij replace this with actual method below once we make network server
function getRandomFaces(done) {
    // const tag = "Faces.getRandomFaces"
    const nodeRootDir = "out/getRandomFaces-2018-06-01-06-15-21-145-iX3xw"
    const imgFilename = "getRandomFaces.jpg"
    const txtFilename = "getRandomFaces.txt"
    const imgFilepath = nodeRootDir + "/" + imgFilename
    const txtFilepath = nodeRootDir + "/" + txtFilename
    let img, encodings
    async.waterfall([
        (done) => {
            FS.readImgFileAsBase64(imgFilepath, done)
        }, (nimg, done) => {
            img = nimg
            FS.readTxtFile(txtFilepath, done)
        }, (txt, done) => {
            encodings = convertTextToEncodings(txt)
            done()
        }
    ], (er) => {
        if (er) done(er)
        else done(null, img, encodings)
    })
}

// poij
function getSimilarFaces(req, res, next) {
    const tag = "Faces.getSimilarFaces"
    const nodeRootDir = "out/getSimilarFaces-" + Time.getTimeYYYYMMDDHHMMSSMS() // out/getSimilarFaces-2018-05-21-01-49-44-862-xDhYP
    const pythonRootDir = "FacesUI/" + nodeRootDir // python cwd is ai/gan/FacesUI/, one above node's root
    const imgFilename = "getSimilarFaces.jpg"
    const txtFilename = "getSimilarFaces.txt"
    const imgFilepath = nodeRootDir + "/" + imgFilename
    const txtFilepath = nodeRootDir + "/" + txtFilename
    let img, encodings
    async.waterfall([
        (done) => {
            // poij
            Faces.makeSimilarFaces(pythonRootDir, imgFilename, txtFilename, done)
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
        // FS.rmdirf(nodeRootDir) // poij
    })
}

// poij remove
// function getSimilarFaces(req, res, next) {
//     // const tag = "Faces.getSimilarFaces"
//     const nodeRootDir = "out/getRandomFaces-2018-06-01-06-15-21-145-iX3xw"
//     const imgFilename = "getRandomFaces.jpg"
//     const txtFilename = "getRandomFaces.txt"
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
//         if (er) next({ status: 500, error: "Cannot get similar faces", er })
//         else res.send({ status: 200, img, encodings })
//     })
// }

/**
 * poij
 * @param {*} done(er, img, encodings)
 */
// function getRandomFaces(done) {
//     // const tag = "Faces.getRandomFaces"
//     const nodeRootDir = "out/getRandomFaces-" + Time.getTimeYYYYMMDDHHMMSSMS() // out/getRandomFaces-2018-05-21-01-49-44-862-xDhYP
//     const pythonRootDir = "FacesUI/" + nodeRootDir // python cwd is ai/gan/FacesUI/, one above node's root
//     const imgFilename = "getRandomFaces.jpg"
//     const txtFilename = "getRandomFaces.txt"
//     const imgFilepath = nodeRootDir + "/" + imgFilename
//     const txtFilepath = nodeRootDir + "/" + txtFilename
//     let img, encodings
//     async.waterfall([
//         (done) => {
//             Faces.makeRandomFaces(pythonRootDir, imgFilename, txtFilename, done)
//         }, (done) => {
//             FS.readImgFileAsBase64(imgFilepath, done)
//         }, (nimg, done) => {
//             img = nimg
//             FS.readTxtFile(txtFilepath, done)
//         }, (txt, done) => {
//             encodings = convertTextToEncodings(txt)
//             done()
//         }
//     ], (er) => {
//         if (er) done(er)
//         else done(null, img, encodings)
//         FS.rmdirf(nodeRootDir)
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

// qwer. Check that entries in the encoding are floats and in the right range [-1, 1].
/**
 * Get a face by its encoding.
 * @param {*} encoding
 * @param {*} done(er, img)
 */
function getFaceByEncoding(encoding, done) {
    // const tag = "Faces.getFaceByEncoding"
    const nodeRootDir = "out/getFaceByEncoding-" + Time.getTimeYYYYMMDDHHMMSSMS() // out/getFaceByEncoding-2018-05-21-01-49-44-862-xDhYP
    const pythonRootDir = "FacesUI/" + nodeRootDir // python cwd is ai/gan/FacesUI/, one above node's root
    const imgFilename = "getFaceByEncoding.jpg"
    const txtFilename = "getFaceByEncoding.txt"
    const imgFilepath = nodeRootDir + "/" + imgFilename
    async.waterfall([
        (done) => {
            Faces.makeFaceByEncoding(encoding, pythonRootDir, imgFilename, txtFilename, done)
        }, (done) => {
            FS.readImgFileAsBase64(imgFilepath, done)
        }, (img, done) => {
            done(null, img)
        }
    ], (er, img) => {
        if (er) done(er)
        else done(null, img)
        // FS.rmdirf(nodeRootDir) // poij
    })
    // poij remove
    // // const tag = "Faces.getRandomFaces"
    // const nodeRootDir = "out/getRandomFaces-2018-06-01-06-15-21-145-iX3xw"
    // const imgFilename = "getRandomFaces.jpg"
    // const imgFilepath = nodeRootDir + "/" + imgFilename
    // async.waterfall([
    //     (done) => {
    //         FS.readImgFileAsBase64(imgFilepath, done)
    //     }, (img, done) => {
    //         done(null, img)
    //     }
    // ], (er, img) => {
    //     if (er) done(er)
    //     else done(null, img)
    // })
}

module.exports = router;
