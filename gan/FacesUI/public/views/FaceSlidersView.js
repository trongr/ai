const FaceSlidersView = (() => {
    const FaceSlidersView = {}

    FaceSlidersView.init = () => {
        FaceSlidersView.createFaceSliders(Conf.CurrentFaceSlidersList)
    }

    /**
     * Init sliders for CurrentFace config
     * @param {String} ID of the parent box to add sliders in.
     */
    FaceSlidersView.createFaceSliders = (ID) => {
        for (let i = 0; i < Conf.NUM_SLIDERS; i++) {
            $("#" + ID).append(MakeFaceSlider(ID, i))
        }
    }

    /**
     * Make a face slider.
     * @param {*} ID of the parent box
     * @param {*} idx
     */
    function MakeFaceSlider(ID, idx) {
        return `<div id="${ID}${idx}" class="CurrentFaceSlider"></div>`
    }

    /**
     * Load encoding into the current face sliders.
     * @param {*} ID of the parent box containing the sliders
     * @param {*} encoding
     */
    FaceSlidersView.loadEncodingIntoCurrentFaceSliders = (ID, encoding) => {
        const tag = "FaceSlidersView.loadEncodingIntoCurrentFaceSliders"
        assert(encoding instanceof Array, `${tag}. Expected encoding to be an Array`)

        for (let i = 0; i < Conf.NUM_SLIDERS; i++) {
            const percent = (encoding[i] + 1) / 2 * 100
            const color = Perc2Color(percent)
            $("#" + ID + i).css("background", color)
        }

        // Resize the slider cells so they form a square.
        const parentWidth = $("#" + ID).width()
        const width = parseInt(parentWidth / Math.sqrt(Conf.NUM_SLIDERS))
        $(".CurrentFaceSlider").width(width).height(width)
    }

    return FaceSlidersView
})();