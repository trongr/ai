const CurrentFaceModel = (() => {
    const CurrentFaceModel = {}

    CurrentFaceModel.encoding = null // encoding for current face

    // poij set default encoding on load.

    /**
     * Save the current face's encoding on face load.
     * @param {*} encoding
     */
    CurrentFaceModel.saveEncoding = (encoding) => {
        CurrentFaceModel.encoding = encoding
    }

    /**
     * Get current encoding.
     */
    CurrentFaceModel.getEncoding = () => {
        return CurrentFaceModel.encoding
    }

    return CurrentFaceModel
})();