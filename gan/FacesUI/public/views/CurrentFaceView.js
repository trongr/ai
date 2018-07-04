const CurrentFaceView = (() => {
    const CurrentFaceView = {}

    /**
     * Load the encoding into the current face, e.g. when user clicks on the
     * grid, or the RENDER button to re-render a slider config; i.e. saves the
     * encoding into the CurrentFaceModel, gets the face render from the server,
     * and load the img into the current face.
     * @param {*} encoding
     */
    CurrentFaceView.getFaceByEncodingAndLoadIntoCurrentFace = async (encoding) => {
        const tag = "CurrentFaceView.getFaceByEncodingAndLoadIntoCurrentFace"
        console.log(tag, encoding)
        const { status, img } = await API.getFaceByEncoding(encoding)
        CurrentFaceModel.saveEncoding(encoding)
        CurrentFaceSlidersView.loadEncodingIntoCurrentFaceSliders(encoding)
        CurrentFaceView.loadEncodingIntoPasteEncodingInput(encoding)
        HistoryView.saveEncodingImg(encoding, img)
        ViewsUtils.loadImgFromBase64("CurrentFace", img)
        ViewsUtils.scrollTo("CurrentFace")
    }

    /**
     * Return the encoding from PasteEncodingInput. Also checks that it matches
     * the format we want: a comma separated list of Conf.NUM_SLIDERS (== 64 for
     * now) floats between -1 and 1.
     */
    CurrentFaceView.getEncodingFromPasteEncodingInput = () => {
        const tag = "CurrentFaceView.getEncodingFromPasteEncodingInput"
        try {
            const encoding = $("#PasteEncodingInput").val().split(",")

            encoding.map(e => {
                e = parseFloat(e)
                assert(e >= -1 && e <= 1, `${tag}: Expected encoding value ${e} to be a float between -1 and 1`)
            })

            assert(encoding.length == Conf.NUM_SLIDERS,
                `${tag}: Expected encoding length ${encoding.length} to be ${Conf.NUM_SLIDERS}`)

            // parseFloat here because we want to print out the original value
            // of the encoding entry. If we'd parseFloat'd before doing the
            // check, the entry would have been a NaN value, and we wouldn't
            // know what the original offending entry was.
            return encoding.map(e => parseFloat(e))
        } catch (e) {
            return null
        }
    }

    /**
     * Loads encoding into PasteEncodingInput so user can copy and save it for
     * later.
     * @param {*} encoding
     */
    CurrentFaceView.loadEncodingIntoPasteEncodingInput = (encoding) => {
        encoding = encoding.map(e => e.toFixed(4))
        $("#PasteEncodingInput").val(encoding.toString())
    }

    return CurrentFaceView
})();