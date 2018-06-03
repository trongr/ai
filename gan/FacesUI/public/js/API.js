const API = (() => {
    const API = {}

    /**
     * Get random faces and their encodings
     */
    API.getRandomFaces = () => {
        return $.ajax({ url: "/faces", type: "GET", data: {} })
    }

    /**
     * Get one face img by its encoding.
     * @param {*} encoding
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

    return API
})();