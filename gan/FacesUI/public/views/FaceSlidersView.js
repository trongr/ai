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
     * @param {*} i index of the slider in the encoding, e.g. 0-63.
     */
    function MakeFaceSlider(ID, i) {
        return `<div id="${ID}${i}" class="FaceSlider"
                    data-encodingparamidx="${i}"></div>`
    }

    /**
     * Load encoding into the current face sliders.
     * @param {*} ID of the parent box containing the sliders
     * @param {*} encoding
     */
    FaceSlidersView.loadEncodingIntoCurrentFaceSliders = (ID, encoding) => {
        const tag = "FaceSlidersView.loadEncodingIntoCurrentFaceSliders"
        assert(encoding instanceof Array, `${tag}. Expected encoding ${encoding} to be an Array`)
        const parentWidth = $("#" + ID).width()
        const width = parseInt(parentWidth / Math.sqrt(Conf.NUM_SLIDERS))
        for (let i = 0; i < Conf.NUM_SLIDERS; i++) {
            FaceSlidersView.setEncodingCellValue(i, ID, encoding[i], width)
        }
    }

    /**
     * ID and i together identifies the Encoding Cell.
     * @param {*} i
     * @param {*} ID of the parent box containing the sliders
     * @param {*} EncodingValue
     * @param {*} width Optional cell width.
     */
    FaceSlidersView.setEncodingCellValue = (i, ID, EncodingValue, width) => {
        const color = Perc2Color((EncodingValue + 1) / 2 * 100)
        const elmt = $(`#${ID}${i}`).css("background", color)
        if (width) elmt.width(width).height(width)
    }

    return FaceSlidersView
})();