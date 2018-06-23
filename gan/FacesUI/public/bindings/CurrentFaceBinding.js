

const CurrentFaceBinding = (() => {
    const CurrentFaceBinding = {}

    CurrentFaceBinding.init = () => {
        CurrentFaceBinding.bindCurrentFaceRenderButton()
        CurrentFaceBinding.bindRenderSimilarFacesButton()
    }

    /**
     * When user clicks the RENDER button, we get the current encoding from the
     * sliders and query the server for its image, and load it into the Current
     * Face.
     */
    CurrentFaceBinding.bindCurrentFaceRenderButton = () => {
        const tag = "CurrentFaceBinding.bindCurrentFaceRenderButton"
        $("#RenderCurrentFaceButton").click(async function (e) {
            try {
                const encoding = CurrentFaceView.getCurrentFaceEncodingFromSliders()
                if (!encoding) return console.error(tag, "No encoding to render")
                CurrentFaceView.getFaceByEncodingAndLoadIntoCurrentFace(encoding)
            } catch (er) {
                console.error(tag, er)
            }
        })
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

    return CurrentFaceBinding
})();