const CurrentFaceModel = (() => {
    const CurrentFaceModel = {}

    CurrentFaceModel.encodings = null // encoding for current face

    /**
     * Save the current face's encoding on face load.
     * @param {*} encoding
     */
    CurrentFaceModel.saveEncoding = (encoding) => {
        CurrentFaceModel.encoding = encoding
    }

    /**
     * Get current encoding. Not used yet because typically we want to get the
     * slider values directly, instead of from this model.
     */
    CurrentFaceModel.getEncoding = () => {
        return CurrentFaceModel.encoding
    }

    return CurrentFaceModel
})();