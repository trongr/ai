const CurrentFaceView = (() => {
    const CurrentFaceView = {}

    const CURRENT_FACE_SLIDER_ID_PREFIX = "CurrentFaceSlider"

    CurrentFaceView.init = () => {
        CurrentFaceView.initCurrentFaceSliders()
    }

    /**
     * Init sliders for CurrentFace config
     */
    CurrentFaceView.initCurrentFaceSliders = () => {
        for (let i = 0; i < Conf.NUM_SLIDERS; i++) {
            const paddedIndex = (i < 10 ? "0" + i : i) // So it aligns a little better.
            $("#CurrentFaceSlidersList").append(paddedIndex + ". \
                <input \
                    id='" + CURRENT_FACE_SLIDER_ID_PREFIX + i + "' \
                    class='CurrentFaceSliderInput' \
                    type='range' min='-1' max='1' step='0.01'> ")
        }
    }

    /**
     * Load encoding into the current face sliders.
     * @param {*} encoding
     */
    CurrentFaceView.loadEncodingIntoCurrentFaceSliders = (encoding) => {
        for (let i = 0; i < Conf.NUM_SLIDERS; i++) {
            $("#" + CURRENT_FACE_SLIDER_ID_PREFIX + i).val(encoding[i])
        }
    }

    /**
     * Get the encoding from the current face sliders
     */
    CurrentFaceView.getCurrentFaceEncodingFromSliders = () => {
        let encoding = []
        for (let i = 0; i < Conf.NUM_SLIDERS; i++) {
            encoding[i] = parseFloat($("#" + CURRENT_FACE_SLIDER_ID_PREFIX + i).val())
        }
        return encoding
    }

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
        CurrentFaceView.loadEncodingIntoCurrentFaceSliders(encoding)
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
                assert(!isNaN(parseFloat(e)), `${tag}: Expected encoding value ${e} to be a float`)
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