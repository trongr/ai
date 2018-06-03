const CurrentFace = (() => {
    const CurrentFace = {}

    CurrentFace.encodings = [] // encoding for current face

    /**
     * Save the current face's encoding on face load.
     * @param {*} encoding
     */
    CurrentFace.saveEncoding = (encoding) => {
        CurrentFace.encoding = encoding
    }

    return CurrentFace
})();