const express = require('express');
const async = require('async');
const Faces = require("../lib/Faces.js")
const FS = require("../core/FS.js")
const Time = require("../core/Time.js")
const router = express.Router();

router.get('/', getRandomFaces)

function getRandomFaces(req, res, next) {
    const tag = "Faces.getRandomFaces"
    const dir = "FacesUI/out/getRandomFaces-" + Time.getTimeYYYYMMDDHHMMSSMS() // ai/gan/FacesUI/out/getRandomFaces-2018-05-21-01-49-44-862-xDhYP
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
        // poij
        // FS.rmdirf(dir)
    })
}

module.exports = router;
