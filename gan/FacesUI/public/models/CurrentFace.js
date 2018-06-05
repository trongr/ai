const CurrentFace = (() => {
    const CurrentFace = {}

    CurrentFace.encodings = null // encoding for current face

    /**
     * Save the current face's encoding on face load.
     * @param {*} encoding
     */
    CurrentFace.saveEncoding = (encoding) => {
        CurrentFace.encoding = encoding
    }

    /**
     * Get current encoding. Not used yet because typically we want to get the
     * slider values directly, instead of from this model.
     */
    CurrentFace.getEncoding = () => {
        return CurrentFace.encoding
    }

    return CurrentFace
})();