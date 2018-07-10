const API = (() => {
  const API = {}

  /**
   * Get random faces and their encodings
   * @return {*} { status, img, encodings }
   */
  API.getRandomFaces = () => {
    return $.ajax({ url: "/faces", type: "GET", data: {} })
  }

  /**
   * Get one face img by its encoding. (Doesn't return encoding cause it's
   * only one image and it's the same encoding as input arg.)
   * @param {*} encoding
   * @return {*} { status, img }
   */
  API.getFaceByEncoding = (encoding) => {
    return $.ajax({
      url: "/faces",
      type: "GET",
      data: {
        encoding,
      },
    })
  }

  /**
   * Get similar faces to encoding. Also returns their encodings
   * @param {*} encoding
   * @param {String} variation Low|Medium|High
   * @return {*} { status, img, encodings }
   */
  API.getSimilarFaces = (encoding, variation) => {
    return $.ajax({
      url: "/faces/similar",
      type: "GET",
      data: {
        encoding,
        variation, // poij add this to API.
      },
    })
  }

  return API
})()
