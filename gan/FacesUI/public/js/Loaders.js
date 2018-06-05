/**
 * Loaders uses API to get data and load them into the Views.
 */
const Loaders = (() => {
    const Loaders = {}

    /**
     * Load the encoding into the current face, e.g. when user clicks on the
     * grid, or the RENDER button to re-render a slider config.
     * @param {*} encoding
     */
    Loaders.loadEncodingIntoCurrentFace = async (encoding) => {
        console.log("DEBUG. Rendering encoding", encoding)
        const { status, img } = await API.getFaceByEncoding(encoding)
        CurrentFace.saveEncoding(encoding)
        Views.loadImgFromBase64("CurrentFace", img)
        Views.loadEncodingIntoCurrentFaceSliders(encoding)
        Views.scrollTo("CurrentFace")
    }

    return Loaders
})();