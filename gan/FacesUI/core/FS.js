/**
 * File system library.
 */
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