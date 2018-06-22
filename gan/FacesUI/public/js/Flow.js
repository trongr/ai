/**
 * Flow uses API to get data and load them into the Views.
 */
const Flow = (() => {
    const Flow = {}

    /**
     * Get random faces from API and load into Random Faces Grid.
     */
    Flow.getRandomFacesAndLoadIntoGrid = async () => {
        const tag = "Flow.getRandomFacesAndLoadIntoGrid"
        const { status, img, encodings } = await API.getRandomFaces()
        FacesGridModel.saveEncodings(encodings,
            Conf.FACES_GRID_NUM_CELLS_ROWS,
            Conf.FACES_GRID_NUM_CELLS_COLS)
        Views.loadImgFromBase64("RandomFacesImg", img)
    }

    /**
     * Load the encoding into the current face, e.g. when user clicks on the
     * grid, or the RENDER button to re-render a slider config.
     * @param {*} encoding
     */
    Flow.getFaceByEncodingAndLoadIntoCurrentFace = async (encoding) => {
        const tag = "Flow.getFaceByEncodingAndLoadIntoCurrentFace"
        console.log(tag, encoding)
        const { status, img } = await API.getFaceByEncoding(encoding)
        CurrentFaceModel.saveEncoding(encoding)
        Views.loadImgFromBase64("CurrentFace", img)
        CurrentFaceView.loadEncodingIntoCurrentFaceSliders(encoding)
        Views.scrollTo("CurrentFace")
    }

    /**
     * Get similar faces to encoding from server and load into Faces Grid.
     * @param {*} encoding
     */
    Flow.getSimilarFacesAndLoadIntoGrid = async (encoding) => {
        const tag = "Flow.getSimilarFacesAndLoadIntoGrid"
        console.log(tag, encoding)
        const { status, img, encodings } = await API.getSimilarFaces(encoding)
        FacesGridModel.saveEncodings(encodings,
            Conf.FACES_GRID_NUM_CELLS_ROWS,
            Conf.FACES_GRID_NUM_CELLS_COLS)
        Views.loadImgFromBase64("RandomFacesImg", img)
        Views.scrollTo("RandomFacesImg")
    }

    return Flow
})();