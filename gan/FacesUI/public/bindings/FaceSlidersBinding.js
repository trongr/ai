

const FaceSlidersBinding = (() => {
    const FaceSlidersBinding = {}

    FaceSlidersBinding.init = () => {
        FaceSlidersBinding.initFaceSliderClick()
    }

    /**
     * poij clicking on the history img doesn't reload the encoding. might not
     * reload the paste encoding input either.
     * poij go in the opposite direction.
     */
    FaceSlidersBinding.initFaceSliderClick = () => {
        const parentID = "CurrentFaceSlidersList"
        $('body').on('click', `#${parentID} .FaceSlider`, function (e) {
            const i = $(this).data("encodingparamidx")
            let EncodingValue = 0.1 + CurrentFaceModel.getEncodingParamAtIndex(i)
            EncodingValue = MathLib.WrapXAroundMinusOneAndOne(EncodingValue)
            CurrentFaceModel.setEncodingParamAtIndex(i, EncodingValue)
            FaceSlidersView.setEncodingCellValue(i, parentID, EncodingValue)
            const encoding = CurrentFaceModel.getEncoding()
            CurrentFaceView.loadEncodingIntoPasteEncodingInput(encoding)
        })
    }

    return FaceSlidersBinding
})();