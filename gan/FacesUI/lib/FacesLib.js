const request = require("request")

const FacesLib = (module.exports = {})

/**
 * Run the neural network and generate random faces
 * @param {*} outputDir The dir to store the output img and text encoding
 * @param {*} imgFilename name of the img output file, including .jpg extension
 * @param {*} txtFilename name of the text encoding file, including .txt extension
 * @param {*} done
 */
FacesLib.GetRandomFaces = (outputDir, imgFilename, txtFilename, done) => {
  const tag = "FacesLib.GetRandomFaces"
  request(
    {
      method: "GET",
      url: "http://localhost:5000/GetRandomFaces",
      qs: { outputDir, imgFilename, txtFilename },
      json: true,
    },
    function(er, res, body) {
      if (er)
        done({
          tag,
          status: 500,
          error: "Server cannot create random faces",
          er,
        })
      else if (body && body.status == 200) done()
      else done({ tag, status: 500, error: "Unknown error", body })
    },
  )
}

/**
 *
 * @param {*} encoding for the face you want similar faces of. A list of float
 * @param {*} std standard deviation for how much to vary from each encoding value
 * @param {*} outputDir
 * @param {*} imgFilename
 * @param {*} txtFilename
 * @param {*} done
 */
FacesLib.GetSimilarFaces = (
  encoding,
  std,
  FreeChannels,
  outputDir,
  imgFilename,
  txtFilename,
  done,
) => {
  const tag = "FacesLib.GetSimilarFaces"
  request(
    {
      method: "GET",
      url: "http://localhost:5000/GetSimilarFaces",
      body: {
        encoding,
        std,
        FreeChannels,
        outputDir,
        imgFilename,
        txtFilename,
      },
      json: true,
    },
    function(er, res, body) {
      if (er)
        done({
          tag,
          status: 500,
          error: "Server cannot create similar faces",
          er,
        })
      else if (body && body.status == 200) done()
      else done({ tag, status: 500, error: "Unknown error", body })
    },
  )
}

/**
 * Run the neural network and generate a single face given its encoding
 * @param {*} encoding Encoding to make the face with
 * @param {*} outputDir The dir to store the output img and text encoding
 * @param {*} imgFilename name of the img output file, including .jpg extension
 * @param {*} txtFilename name of the text encoding file, including .txt extension
 * @param {*} done
 */
FacesLib.GetFaceByEncoding = (
  encoding,
  outputDir,
  imgFilename,
  txtFilename,
  done,
) => {
  const tag = "FacesLib.GetFaceByEncoding"
  request(
    {
      method: "GET",
      url: "http://localhost:5000/GetFaceByEncoding",
      body: { encoding, outputDir, imgFilename, txtFilename },
      json: true,
    },
    function(er, res, body) {
      if (er)
        done({
          tag,
          status: 500,
          error: "Server cannot create face by encoding",
          er,
        })
      else if (body && body.status == 200) done()
      else done({ tag, status: 500, error: "Unknown error", body })
    },
  )
}
