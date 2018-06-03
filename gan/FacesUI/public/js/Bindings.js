const Bindings = (() => {
    const Bindings = {}

    const FACES_GRID_NUM_CELLS_COLS = 10
    const FACES_GRID_NUM_CELLS_ROWS = 10

    /**
     * Bind all events
     */
    Bindings.init = () => {
        bindGenerateRandomFacesButton()
        bindRandomFacesGridClick()
    }

    /**
     * Generate random faces
     */
    function bindGenerateRandomFacesButton() {
        $("#GenerateRandomFacesButton").click(async function (e) {
            try {
                const { status, img, encodings } = await API.getRandomFaces()
                Views.loadImgFromBase64("RandomFacesImg", img)
                RandomFacesGrid.saveEncodings(encodings,
                    FACES_GRID_NUM_CELLS_ROWS,
                    FACES_GRID_NUM_CELLS_COLS)
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
    function bindRandomFacesGridClick() {
        $("#RandomFacesImg").click(async function (e) {
            // TODO. Let the server tell you how many cells there are instead of
            // hardcoding it here
            const cellWidth = $(this).width() / FACES_GRID_NUM_CELLS_COLS
            const cellHeight = $(this).height() / FACES_GRID_NUM_CELLS_ROWS
            const rawX = e.pageX - $(this).offset().left
            const rawY = e.pageY - $(this).offset().top
            const j = parseInt(rawX / cellWidth)
            const i = parseInt(rawY / cellHeight)
            const encoding = RandomFacesGrid.getEncoding(i, j)
            console.log("DEBUG. Clicking cell", i, j)
            try {
                const { status, img } = await API.getFaceByEncoding(encoding)
                // poij scroll down to current face
                console.log("poij get face by encoding response", status, img)
                Views.loadImgFromBase64("CurrentFace", img)
                CurrentFace.saveEncoding(encoding)
            } catch (er) {
                console.error("bindRandomFacesGridClick", er)
            }
        })
    }

    return Bindings
})();