const Bindings = (() => {
    const Bindings = {}

    /**
     * Bind all events
     */
    Bindings.init = () => {
        bindGenerateRandomFacesButton()
        bindFacesGridClick()
        bindCurrentFaceRenderButton()
        bindRenderSimilarFacesButton()
    }

    /**
     * Generate random faces
     */
    function bindGenerateRandomFacesButton() {
        $("#GenerateRandomFacesButton").click(async function (e) {
            try {
                Flow.getRandomFacesAndLoadIntoGrid()
            } catch (er) {
                console.error("bindGenerateRandomFacesButton", er)
            }
        })
    }

    /**
     * When user clicks on the grid of random faces, we want to get the cell
     * location, get its encoding, and make a request to the server to render
     * that face, and finally load it into the current face.
     */
    function bindFacesGridClick() {
        $("#RandomFacesImg").click(async function (e) {
            try {
                // TODO. Let the server tell you how many cells there are instead of
                // hardcoding it here
                const cellWidth = $(this).width() / Conf.FACES_GRID_NUM_CELLS_COLS
                const cellHeight = $(this).height() / Conf.FACES_GRID_NUM_CELLS_ROWS
                const rawX = e.pageX - $(this).offset().left
                const rawY = e.pageY - $(this).offset().top
                const j = parseInt(rawX / cellWidth)
                const i = parseInt(rawY / cellHeight)
                const encoding = FacesGridModel.getEncoding(i, j)
                Flow.getFaceByEncodingAndLoadIntoCurrentFace(encoding)
                console.log("DEBUG. Clicking cell", i, j)
            } catch (er) {
                console.error("bindFacesGridClick", er)
            }
        })
    }

    /**
     * When user clicks the RENDER button, we get the current encoding from the
     * sliders and query the server for its image, and load it into the Current
     * Face.
     */
    function bindCurrentFaceRenderButton() {
        $("#RenderCurrentFaceButton").click(async function (e) {
            try {
                const encoding = CurrentFaceView.getCurrentFaceEncodingFromSliders()
                if (!encoding) return console.error("No encoding to render")
                Flow.getFaceByEncodingAndLoadIntoCurrentFace(encoding)
            } catch (er) {
                console.error("bindCurrentFaceRenderButton", er)
            }
        })
    }

    /**
     * When user clicks SIMILAR FACES button, we query the server for faces
     * similar to the current encoding, and load it into the Faces Grid.
     */
    function bindRenderSimilarFacesButton() {
        $("#RenderSimilarFacesButton").click(async function (e) {
            try {
                const encoding = CurrentFaceView.getCurrentFaceEncodingFromSliders()
                if (!encoding) return console.error("No encoding to render")
                Flow.getSimilarFacesAndLoadIntoGrid(encoding)
            } catch (er) {
                console.error("bindRenderSimilarFacesButton", er)
            }
        })
    }

    return Bindings
})();