const CurrentFaceSlidersView = (() => {
    const CurrentFaceSlidersView = {}

    const CURRENT_FACE_SLIDER_ID_PREFIX = "CurrentFaceSlider"

    CurrentFaceSlidersView.init = () => {
        CurrentFaceSlidersView.initCurrentFaceSliders()
    }

    /**
     * Init sliders for CurrentFace config
     */
    CurrentFaceSlidersView.initCurrentFaceSliders = () => {
        for (let i = 0; i < Conf.NUM_SLIDERS; i++) {
            $("#CurrentFaceSlidersList").append(MakeFaceSlider(i))
        }
    }

    /**
     * Make a face slider.
     * @param {*} idx
     */
    function MakeFaceSlider(idx) {
        return `<div id="${CURRENT_FACE_SLIDER_ID_PREFIX}${idx}" \
                    class="CurrentFaceSlider"></div>`
    }

    /**
     * Load encoding into the current face sliders.
     * @param {*} encoding
     */
    CurrentFaceSlidersView.loadEncodingIntoCurrentFaceSliders = (encoding) => {
        const tag = "CurrentFaceSlidersView.loadEncodingIntoCurrentFaceSliders"
        assert(encoding instanceof Array, `${tag}. Expected encoding to be an Array`)

        for (let i = 0; i < Conf.NUM_SLIDERS; i++) {
            const percent = (encoding[i] + 1) / 2 * 100
            const color = Perc2Color(percent)
            $("#" + CURRENT_FACE_SLIDER_ID_PREFIX + i).css("background", color)
        }

        // Resize the slider cells so they form a square.
        const parentWidth = $("#CurrentFaceSlidersList").width()
        const width = parseInt(parentWidth / Math.sqrt(Conf.NUM_SLIDERS))
        $(".CurrentFaceSlider").width(width).height(width)
    }

    return CurrentFaceSlidersView
})();