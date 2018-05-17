const { spawn } = require('child_process')

const Faces = module.exports = {}

Faces.makeRandomFaces = (done) => {
    const tag = "Faces.makeRandomFaces"
    // By default the cwd is the directory you ran node from i.e.
    // ai/gan/FacesUI/, so cwd:.. sets cwd to parent dir ai/gan/.
    const child = spawn('bash', ["RunRandomFaces.sh"], { cwd: ".." })
    child.on('error', (er) => console.error(tag, er))
    child.stdout.on('data', (data) => console.log(tag, data.toString()))
    child.stderr.on('data', (data) => console.error(tag, data.toString()))
    child.on('exit', (code, signal) => code == 0 ? done() : done({ code, signal }))
}