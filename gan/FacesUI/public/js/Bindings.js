const Bindings = (() => {
    const Bindings = {}

    /**
     * Bind all events
     */
    Bindings.init = () => {
        bindGenerateRandomFacesButton()
    }

    /**
     * Generate random faces
     */
    function bindGenerateRandomFacesButton() {
        $("#GenerateRandomFaces").click(async e => {
            try {
                const { status, img, encodings } = await API.getRandomFaces()
                Views.loadImgFromBase64("RandomFacesImg", img)
            } catch (er) {
                console.error(JSON.stringify(er, null, 4))
            }
        })
    }

    return Bindings
})();