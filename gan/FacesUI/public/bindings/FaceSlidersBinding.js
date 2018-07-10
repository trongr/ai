const FaceSlidersBinding = (() => {
  const FaceSlidersBinding = {}

  FaceSlidersBinding.init = () => {
    FaceSlidersBinding.initFaceSliderClick()
  }

  FaceSlidersBinding.initFaceSliderClick = () => {
    const parentID = "CurrentFaceSlidersList"
    $("body").on("click", `#${parentID} .FaceSlider`, function(e) {
      const i = $(this).data("encodingparamidx")

      let EncodingValue = CurrentFaceModel.getEncodingParamAtIndex(i)
      // Increase on click, decrease on ALT + click.
      EncodingValue = EncodingValue + (e.altKey ? 0.2 : -0.2)
      EncodingValue = MathLib.WrapXAroundMinusOneAndOne(EncodingValue)

      CurrentFaceModel.setEncodingParamAtIndex(i, EncodingValue)
      FaceSlidersView.setEncodingCellValue(i, parentID, EncodingValue)

      const encoding = CurrentFaceModel.getEncoding()
      CurrentFaceView.loadEncodingIntoPasteEncodingInput(encoding)
    })
  }

  return FaceSlidersBinding
})()
