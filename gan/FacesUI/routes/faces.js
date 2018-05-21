const express = require('express');
const async = require('async');
const Faces = require("../lib/Faces.js")
const router = express.Router();

router.get('/', getRandomFaces)

function getRandomFaces(req, res, next) {
    const tag = "Faces.getRandomFaces"
    // poij add date time for dir too, so you can just remove it at the end of the request.
    const dir = "FacesUI/out/" // ai/gan/FacesUI/out/
    // poij add date time for img filenames
    const imgFilename = "getRandomFaces.jpg"
    const txtFilename = "getRandomFaces.txt"
    async.waterfall([
        (done) => {
            Faces.makeRandomFaces(dir, imgFilename, txtFilename, (er) => done(er))
        },
        (done) => {
            // poij read img and text files and return to client
            done()
        },
    ], (er) => {
        if (er) next({ status: 500, error: "Cannot create random faces", er })
        else res.send({ status: 200 })
        // poij remove the output dir
    })
}

module.exports = router;
