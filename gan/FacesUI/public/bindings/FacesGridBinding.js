const FacesGridBinding = (() => {
    const FacesGridBinding = {}

    FacesGridBinding.init = () => {
        FacesGridBinding.bindGenerateRandomFacesButton()
        FacesGridBinding.bindFacesGridClick()
    }

    /**
     * Generate random faces
     */
    FacesGridBinding.bindGenerateRandomFacesButton = () => {
        const tag = "FacesGridBinding.bindGenerateRandomFacesButton"
        $("#GenerateRandomFacesButton").click(async function (e) {
            try {
                FacesGridView.getRandomFacesAndLoadIntoGrid()
            } catch (er) {
                console.error(tag, er)
            }
        })
    }

    /**
     * When user clicks on the grid of random faces, we want to get the cell
     * location, get its encoding, and make a request to the server to render
     * that face, and finally load it into the current face.
     */
    FacesGridBinding.bindFacesGridClick = () => {
        const tag = "FacesGridBinding.bindFacesGridClick"
        $("#FacesGridImg").click(async function (e) {
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
                CurrentFaceView.getFaceByEncodingAndLoadIntoCurrentFace(encoding)
                console.log(tag, "DEBUG. Clicking cell", i, j)
            } catch (er) {
                console.error(tag, er)
            }
        })
    }

    return FacesGridBinding
})();