/**
 * File system library.
 */
const fs = require("fs")
const path = require("path")
const rimraf = require("rimraf")
const FS = module.exports = {}

/**
 * Creates a dir, does nothing if already exists, like mkdir -p.
 * @param {*} dirname
 * @param {*} done
 */
FS.mkdirp = (dirname, done) => {
    // TODO
}

/**
 * Force removes dir, like rm -rf
 * @param {*} dirname
 * @param {*} done
 */
FS.rmdirf = (dirname, done) => {
    rimraf(dirname, er => done ? done(er) : console.error("FS.rmdirf. Cannot remove dir", er))
}

/**
 * Reads image from file and returns it in base64
 * @param {*} filepath points to an image file, jpg or png.
 * @param {*} done
 */
FS.readImgFileAsBase64 = (filepath, done) => {
    fs.readFile(filepath, (er, data) => {
        if (er) return done(er)
        let extension = path.extname(filepath).split('.').pop()
        let base64Image = new Buffer(data, 'binary').toString('base64')
        let imgSrcString = `data:image/${extension};base64,${base64Image}`
        done(null, imgSrcString)
    })
}

/**
 * Read text file and return string content
 * @param {*} filepath
 * @param {*} done
 */
FS.readTxtFile = (filepath, done) => {
    fs.readFile(filepath, "utf8", (er, data) => {
        if (er) return done(er)
        done(null, data)
    })
}