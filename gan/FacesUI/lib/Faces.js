const request = require("request")

const Faces = module.exports = {}

/**
 * Run the neural network and generate random faces
 * @param {*} outputDir The dir to store the output img and text encoding
 * @param {*} imgFilename name of the img output file, including .jpg extension
 * @param {*} txtFilename name of the text encoding file, including .txt extension
 * @param {*} done
 */
Faces.GetRandomFaces = (outputDir, imgFilename, txtFilename, done) => {
    const tag = "Faces.GetRandomFaces"
    request({
        method: "GET",
        url: "http://localhost:5000/GetRandomFaces",
        qs: { outputDir, imgFilename, txtFilename },
        json: true
    }, function (er, res, body) {
        if (er) done({ tag, status: 500, error: "Server cannot create random faces", er })
        else if (body && body.status == 200) done()
        else done({ tag, status: 500, error: "Unknown error", body })
    })
}

/**
 *
 * @param {*} encoding for the face you want similar faces of. A list of float
 * values
 * @param {*} outputDir
 * @param {*} imgFilename
 * @param {*} txtFilename
 * @param {*} done
 */
Faces.GetSimilarFaces = (encoding, outputDir, imgFilename, txtFilename, done) => {
    const tag = "Faces.GetSimilarFaces"
    request({
        method: "GET",
        url: "http://localhost:5000/GetSimilarFaces",
        body: { encoding, outputDir, imgFilename, txtFilename },
        json: true
    }, function (er, res, body) {
        if (er) done({ tag, status: 500, error: "Server cannot create similar faces", er })
        else if (body && body.status == 200) done()
        else done({ tag, status: 500, error: "Unknown error", body })
    })
}

/**
 * Run the neural network and generate a single face given its encoding
 * @param {*} encoding Encoding to make the face with
 * @param {*} outputDir The dir to store the output img and text encoding
 * @param {*} imgFilename name of the img output file, including .jpg extension
 * @param {*} txtFilename name of the text encoding file, including .txt extension
 * @param {*} done
 */
Faces.GetFaceByEncoding = (encoding, outputDir, imgFilename, txtFilename, done) => {
    const tag = "Faces.GetFaceByEncoding"
    request({
        method: "GET",
        url: "http://localhost:5000/GetFaceByEncoding",
        body: { encoding, outputDir, imgFilename, txtFilename },
        json: true
    }, function (er, res, body) {
        if (er) done({ tag, status: 500, error: "Server cannot create face by encoding", er })
        else if (body && body.status == 200) done()
        else done({ tag, status: 500, error: "Unknown error", body })
    })
}