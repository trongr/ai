

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
                const encoding = CurrentFaceView.getCurrentFaceEncodingFromSliders()
                if (!encoding) return console.error(tag, "No encoding to render")
                FacesGridView.getSimilarFacesAndLoadIntoGrid(encoding)
            } catch (er) {
                console.error(tag, er)
            }
        })
    }

    /**
     * When user clicks RenderEncodingButton, we take the list of floats in
     * PasteEncodingInput (if any) and loads it into the sliders.
     */
    CurrentFaceBinding.bindRenderEncodingButton = () => {
        const tag = "CurrentFaceBinding.bindRenderEncodingButton"
        $("#RenderEncodingButton").click(async function (e) {
            try {
                const encoding = CurrentFaceView.getEncodingFromPasteEncodingInput()
                if (!encoding) return // This is a user error.
                CurrentFaceView.getFaceByEncodingAndLoadIntoCurrentFace(encoding)
            } catch (er) {
                console.error(tag, er)
            }
        })
    }

    return CurrentFaceBinding
})();