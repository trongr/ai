const API = (() => {
    const API = {}

    /**
     * Get random faces and their encodings
     * @returns { status, img, encodings }
     */
    API.getRandomFaces = () => {
        return $.ajax({ url: "/faces", type: "GET", data: {} })
    }

    /**
     * Get one face img by its encoding. (Doesn't return encoding cause it's
     * only one image and it's the same encoding as input arg.)
     * @param {*} encoding
     * @returns { status, img }
     */
    API.getFaceByEncoding = (encoding) => {
        return $.ajax({
            url: "/faces",
            type: "GET",
            data: {
                encoding
            }
        })
    }

    /**
     * Get similar faces to encoding. Also returns their encodings
     * @param {*} encoding
     * @returns { status, img, encodings }
     */
    API.getSimilarFaces = (encoding) => {
        return $.ajax({
            url: "/faces/similar",
            type: "GET",
            data: {
                encoding
            }
        })
    }

    return API
})();