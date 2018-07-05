const CurrentFaceModel = (() => {
    const CurrentFaceModel = {}

    CurrentFaceModel.encoding = [] // encoding for current face

    CurrentFaceModel.init = () => {
        const encoding = CurrentFaceView.getEncodingFromPasteEncodingInput()
        // Use existing values in PasteEncodingInput if any, OTW default to 0's.
        if (encoding) {
            CurrentFaceModel.saveEncoding(encoding)
            FaceSlidersView.loadEncodingIntoCurrentFaceSliders(Conf.CurrentFaceSlidersList, encoding)
        } else {
            const encoding = []
            for (let i = 0; i < Conf.NUM_SLIDERS; i++) { encoding.push(0) }
            CurrentFaceModel.saveEncoding(encoding)
            FaceSlidersView.loadEncodingIntoCurrentFaceSliders(Conf.CurrentFaceSlidersList, encoding)
            CurrentFaceView.loadEncodingIntoPasteEncodingInput(encoding)
        }
    }

    /**
     * Save the current face's encoding on face load. Clone deep in case caller
     * modifies it on the outside.
     * @param {*} encoding
     */
    CurrentFaceModel.saveEncoding = (encoding) => {
        CurrentFaceModel.encoding = _.cloneDeep(encoding)
    }

    /**
     * Get current encoding. Clone deep in case caller modifies it on the
     * outside.
     */
    CurrentFaceModel.getEncoding = () => {
        return _.cloneDeep(CurrentFaceModel.encoding)
    }

    /**
     * Get the encoding value at index
     * @param {*} i
     */
    CurrentFaceModel.getEncodingParamAtIndex = (i) => {
        return CurrentFaceModel.encoding[i]
    }

    /**
     * Set
     * @param {*} i
     * @param {*} v
     */
    CurrentFaceModel.setEncodingParamAtIndex = (i, v) => {
        CurrentFaceModel.encoding[i] = v
    }

    return CurrentFaceModel
})();