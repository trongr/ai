const RandomFacesGrid = (() => {
    const RandomFacesGrid = {}

    let encodings = [] // encodings for random faces

    RandomFacesGrid.saveEncodings = (nencodings) => {
        encodings = nencodings
    }

    return RandomFacesGrid
})();