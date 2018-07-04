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
     *
     * @param {*} ID
     * @param {*} encoding
     */
    FaceSlidersView.createFaceSlidersWithEncoding = (ID, encoding) => {
        FaceSlidersView.createFaceSliders(ID)
        FaceSlidersView.loadEncodingIntoCurrentFaceSliders(ID, encoding)
    }

    /**
     * Make a face slider.
     * @param {*} ID of the parent box
     * @param {*} idx index of the slider in the encoding, e.g. 0-63.
     */
    function MakeFaceSlider(ID, idx) {
        return `<div id="${ID}${idx}" class="FaceSlider"></div>`
    }

    /**
     * Load encoding into the current face sliders.
     * @param {*} ID of the parent box containing the sliders
     * @param {*} encoding
     */
    FaceSlidersView.loadEncodingIntoCurrentFaceSliders = (ID, encoding) => {
        const tag = "FaceSlidersView.loadEncodingIntoCurrentFaceSliders"
        assert(encoding instanceof Array,
            `${tag}. Expected encoding ${encoding} to be an Array`)
        const parentWidth = $("#" + ID).width()
        const width = parseInt(parentWidth / Math.sqrt(Conf.NUM_SLIDERS))
        for (let i = 0; i < Conf.NUM_SLIDERS; i++) {
            const color = Perc2Color((encoding[i] + 1) / 2 * 100)
            $(`#${ID}${i}`).css("background", color).width(width).height(width)
        }
    }

    return FaceSlidersView
})();