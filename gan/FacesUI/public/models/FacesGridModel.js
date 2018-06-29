const FacesGridModel = (() => {
    const FacesGridModel = {}

    FacesGridModel.encodings = [] // encodings for random faces

    /**
     * Save new encoding when user loads a new set of random faces into the
     * grid. Also reshapes the encoding from a flat list of say 100 encodings
     * into a grid of size NUM_CELLS_ROWS by NUM_CELLS_COLS.
     * @param {*} nencodings
     * @param {*} NUM_CELLS_ROWS How many cells per row
     * @param {*} NUM_CELLS_COLS How many cells per column
     */
    FacesGridModel.saveEncodings = (nencodings, NUM_CELLS_ROWS = 10, NUM_CELLS_COLS = 10) => {
        const tag = "FacesGridModel.saveEncodings"
        assert(nencodings.length == NUM_CELLS_ROWS * NUM_CELLS_COLS,
            `${tag}: Random faces encoding length ${nencodings.length} doesn't match expected number of rows ${NUM_CELLS_ROWS} and columns ${NUM_CELLS_COLS}`)
        let encodings = []
        nencodings.map((e, idx) => {
            const i = parseInt(idx / NUM_CELLS_ROWS) // However many multiples is the row number i
            const j = idx - (i * NUM_CELLS_ROWS) // Whatever remains is the column number j
            if (j == 0) encodings.push([]) // Create a new row
            encodings[i][j] = e
        })
        FacesGridModel.encodings = encodings
    }

    /**
     * Get encoding at position i, j. E.g. user has clicked on a cell in the
     * random faces grid, and we want to load it into the current face UI.
     * @param {*} i cell row
     * @param {*} j cell column
     */
    FacesGridModel.getEncoding = (i, j) => {
        return FacesGridModel.encodings[i][j]
    }

    return FacesGridModel
})();