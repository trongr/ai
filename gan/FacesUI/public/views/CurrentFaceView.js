const CurrentFaceView = (() => {
    const CurrentFaceView = {}

    const NUM_SLIDERS = 64 // TODO. Right now we're using 64 params for the z-encoding
    const CURRENT_FACE_SLIDER_ID_PREFIX = "CurrentFaceSlider"

    CurrentFaceView.init = () => {
        CurrentFaceView.initCurrentFaceSliders()
    }

    /**
     * Init sliders for CurrentFace config
     */
    CurrentFaceView.initCurrentFaceSliders = () => {
        for (let i = 0; i < NUM_SLIDERS; i++) {
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
        for (let i = 0; i < NUM_SLIDERS; i++) {
            $("#" + CURRENT_FACE_SLIDER_ID_PREFIX + i).val(encoding[i])
        }
    }

    /**
     * Get the encoding from the current face sliders
     */
    CurrentFaceView.getCurrentFaceEncodingFromSliders = () => {
        let encoding = []
        for (let i = 0; i < NUM_SLIDERS; i++) {
            encoding[i] = parseFloat($("#" + CURRENT_FACE_SLIDER_ID_PREFIX + i).val())
        }
        return encoding
    }

    return CurrentFaceView
})();