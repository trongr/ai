const CurrentFaceModel = (() => {
    const CurrentFaceModel = {}

    CurrentFaceModel.encoding = [] // encoding for current face

    CurrentFaceModel.init = () => {
        const encoding = CurrentFaceView.getEncodingFromPasteEncodingInput()
        // Use existing values in PasteEncodingInput if any, OTW default to 0's.
        if (encoding) {
            CurrentFaceModel.saveEncoding(encoding)
            CurrentFaceSlidersView.loadEncodingIntoCurrentFaceSliders(encoding)
        } else {
            const encoding = []
            for (let i = 0; i < Conf.NUM_SLIDERS; i++) { encoding.push(0) }
            CurrentFaceModel.saveEncoding(encoding)
            CurrentFaceView.loadEncodingIntoCurrentFaceSliders(encoding)
            CurrentFaceView.loadEncodingIntoPasteEncodingInput(encoding)
        }
    }
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