const { spawn } = require('child_process')

const Faces = module.exports = {}

/**
 * Run the neural network and generate random faces
 * @param {*} dir The dir to store the output img and text encoding
 * @param {*} imgFilename name of the img output file, including .jpg extension
 * @param {*} txtFilename name of the text encoding file, including .txt extension
 * @param {*} done
 */
// poij call flask server instead of running standalone tensorflow model
Faces.makeRandomFaces = (dir, imgFilename, txtFilename, done) => {
    const tag = "Faces.makeRandomFaces"
    // poij Copy this to NOTES.
    // By default the cwd is the directory you ran node from i.e.
    // ai/gan/FacesUI/, so cwd:.. sets cwd to parent dir ai/gan/.
    const child = spawn('bash',
        ["RunRandomFaces.sh", dir, imgFilename, txtFilename],
        { cwd: ".." })
    child.on('error', (er) => console.error(tag, er))
    child.stdout.on('data', (data) => console.log(tag, data.toString()))
    child.stderr.on('data', (data) => console.error(tag, data.toString()))
    child.on('exit', (code, signal) => code == 0 ? done() : done({ tag, code, signal }))
}

/**
 *
 * @param {*} encoding
 * @param {*} dir
 * @param {*} imgFilename
 * @param {*} txtFilename
 * @param {*} done
 */
Faces.makeSimilarFaces = (encoding, dir, imgFilename, txtFilename, done) => {
    const tag = "Faces.makeSimilarFaces"
    // By default the cwd is the directory you ran node from i.e.
    // ai/gan/FacesUI/, so cwd:.. sets cwd to parent dir ai/gan/.
    const child = spawn('bash',
        ["RunMakeSimilarFaces.sh", encoding, dir, imgFilename, txtFilename],
        { cwd: ".." })
    child.on('error', (er) => console.error(tag, er))
    child.stdout.on('data', (data) => console.log(tag, data.toString()))
    child.stderr.on('data', (data) => console.error(tag, data.toString()))
    child.on('exit', (code, signal) => code == 0 ? done() : done({ tag, code, signal }))
}

/**
 * Run the neural network and generate a single face given its encoding
 * @param {*} dir The dir to store the output img and text encoding
 * @param {*} imgFilename name of the img output file, including .jpg extension
 * @param {*} txtFilename name of the text encoding file, including .txt extension
 * @param {*} done
 */
Faces.makeFaceByEncoding = (encoding, dir, imgFilename, txtFilename, done) => {
    const tag = "Faces.makeFaceByEncoding"
    // By default the cwd is the directory you ran node from i.e.
    // ai/gan/FacesUI/, so cwd:.. sets cwd to parent dir ai/gan/.
    const child = spawn('bash',
        ["RunMakeFaceByEncoding.sh", encoding, dir, imgFilename, txtFilename],
        { cwd: ".." })
    child.on('error', (er) => console.error(tag, er))
    child.stdout.on('data', (data) => console.log(tag, data.toString()))
    child.stderr.on('data', (data) => console.error(tag, data.toString()))
    child.on('exit', (code, signal) => code == 0 ? done() : done({ tag, code, signal }))
}