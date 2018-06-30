

const CurrentFaceBinding = (() => {
    const CurrentFaceBinding = {}

    CurrentFaceBinding.init = () => {
        CurrentFaceBinding.bindRenderSimilarFacesButton()
        CurrentFaceBinding.bindRenderEncodingButton()
        CurrentFaceBinding.bindPasteEncodingInput()
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
                FacesGridView.getSimilarFacesAndLoadIntoGrid(encoding)
            } catch (er) {
                console.error(tag, er)
            }
        })
    }

    CurrentFaceBinding.bindPasteEncodingInput = () => {
        const tag = "CurrentFaceBinding.bindPasteEncodingInput"
        $("#PasteEncodingInput").on("input", function (e) {
            const encoding = CurrentFaceView.getEncodingFromPasteEncodingInput()
            if (encoding) {
                console.log(tag, "Updating:", encoding)
                CurrentFaceModel.saveEncoding(encoding)
                CurrentFaceView.loadEncodingIntoCurrentFaceSlidersAndPasteEncodingInput(encoding)
            } else console.error(tag, "Invalid encoding")
        })
    }

    /**
     * When user clicks RenderEncodingButton, we take the encoding in the
     * CurrentFaceModel, request it from the server, and load it into the
     * current face.
     */
    CurrentFaceBinding.bindRenderEncodingButton = () => {
        const tag = "CurrentFaceBinding.bindRenderEncodingButton"
        $("#RenderEncodingButton").click(async function (e) {
            try {
                const encoding = CurrentFaceModel.getEncoding()
                CurrentFaceView.getFaceByEncodingAndLoadIntoCurrentFace(encoding)
            } catch (er) {
                console.error(tag, er)
            }
        })
    }

    return CurrentFaceBinding
})();