const Views = (() => {
    const Views = {}

    /**
     *
     * @param {*} id The id of the image element, doesn't contain #.
     * @param {*} img The base64 content of the image.
     */
    Views.loadImgFromBase64 = (id, img) => {
        document.getElementById(id).src = img
    }

    return Views
})();