const ViewsUtils = (() => {
    const ViewsUtils = {}

    /**
     * Load the random faces img into the random faces grid.
     * @param {*} id The id of the image element, doesn't contain #.
     * @param {*} img The base64 content of the image.
     */
    ViewsUtils.loadImgFromBase64 = (id, img) => {
        document.getElementById(id).src = img
    }

    /**
     * Smooth scrolling to element
     * @param {*} elementID
     */
    ViewsUtils.scrollTo = (elementID) => {
        $('html,body').animate({
            scrollTop: $("#" + elementID).offset().top
        })
    }

    return ViewsUtils
})();
