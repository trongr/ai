const express = require('express');
const router = express.Router();
const Faces = require("../lib/Faces.js")

router.get('/', function (req, res, next) {
    Faces.makeRandomFaces((er, re) => {
        if (er) next({ status: 500, error: "Cannot create random faces", er, re })
        else res.send({ status: 200, re })
    })
});

module.exports = router;
