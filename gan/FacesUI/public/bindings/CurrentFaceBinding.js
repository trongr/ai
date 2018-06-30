

const CurrentFaceBinding = (() => {
    const CurrentFaceBinding = {}

    CurrentFaceBinding.init = () => {
        CurrentFaceBinding.bindRenderSimilarFacesButton()
        CurrentFaceBinding.bindRenderEncodingButton()
    }

    /**
     * When user clicks SIMILAR FACES button, we query the server for faces
     * similar to the current encoding, and load it into the Faces Grid.
     */
    CurrentFaceBinding.bindRenderSimilarFacesButton = () => {
        const tag = "CurrentFaceBinding.bindRenderSimilarFacesButton"
        $("#RenderSimilarFacesButton").click(async function (e) {
            try {
                const encoding = CurrentFaceModel.getEncoding()
                // poij set default encoding on load
                if (!encoding) return console.error(tag, "No encoding to render")
                FacesGridView.getSimilarFacesAndLoadIntoGrid(encoding)
            } catch (er) {
                console.error(tag, er)
            }
        })
    }

    // poij add binding for pasteencodinginput change, and update encoding
    // model.
    /**
     * When user clicks RenderEncodingButton, we take the list of floats in
     * PasteEncodingInput (if any) and loads it into the sliders.
     */
    CurrentFaceBinding.bindRenderEncodingButton = () => {
        const tag = "CurrentFaceBinding.bindRenderEncodingButton"
        $("#RenderEncodingButton").click(async function (e) {
            try {
                const encoding = CurrentFaceModel.getEncoding()
                // poij set default encoding on load
                if (!encoding) return console.error(tag, "No encoding to render")
                CurrentFaceView.getFaceByEncodingAndLoadIntoCurrentFace(encoding)
            } catch (er) {
                console.error(tag, er)
            }
        })
    }

    return CurrentFaceBinding
})();