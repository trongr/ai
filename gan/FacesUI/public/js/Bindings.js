const Bindings = (() => {
    const Bindings = {}

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
        $("#GenerateRandomFaces").click(async e => {
            try {
                const { status, img, encodings } = await API.getRandomFaces()
                Views.loadImgFromBase64("RandomFacesImg", img)
                RandomFaces.saveEncodings(encodings)
            } catch (er) {
                console.error(JSON.stringify(er, null, 4))
            }
        })
    }

    /**
     * When user clicks on the grid of random faces, we want to get the cell
     * location, get its encoding, and make a request to the server to render
     * that face, and finally load it into the current face.
     */
    function bindRandomFacesGridClick() {

    }

    return Bindings
})();