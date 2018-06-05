const Views = (() => {
    const Views = {}

    const NUM_SLIDERS = 64 // TODO. Right now we're using 64 params for the z-encoding
    const CURRENT_FACE_SLIDER_ID_PREFIX = "CurrentFaceSlider"

    Views.init = () => {
        Views.initCurrentFaceSliders()
    }

    /**
     * Load the random faces img into the random faces grid.
     * @param {*} id The id of the image element, doesn't contain #.
     * @param {*} img The base64 content of the image.
     */
    Views.loadImgFromBase64 = (id, img) => {
        document.getElementById(id).src = img
    }

    // poij set these values when we load a new face.
    Views.initCurrentFaceSliders = () => {
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
     * Smooth scrolling to element
     * @param {*} elementID
     */
    Views.scrollTo = (elementID) => {
        $('html,body').animate({
            scrollTop: $("#" + elementID).offset().top
        })
    }

    /**
     * Load encoding into the current face sliders.
     * @param {*} encoding
     */
    Views.loadEncodingIntoCurrentFaceSliders = (encoding) => {
        for (let i = 0; i < NUM_SLIDERS; i++) {
            $("#" + CURRENT_FACE_SLIDER_ID_PREFIX + i).val(encoding[i])
        }
    }

    /**
     * Get the encoding from the current face sliders
     */
    Views.getCurrentFaceEncodingFromSliders = () => {
        let encoding = []
        for (let i = 0; i < NUM_SLIDERS; i++) {
            encoding[i] = parseFloat($("#" + CURRENT_FACE_SLIDER_ID_PREFIX + i).val())
        }
        return encoding
    }

    return Views
})();