const FacesGridView = (() => {
  const FacesGridView = {}

  /**
   * Get random faces from API and load into Faces Grid.
   */
  FacesGridView.getRandomFacesAndLoadIntoGrid = async () => {
    const { img, encodings } = await API.getRandomFaces()
    FacesGridModel.saveEncodings(
      encodings,
      Conf.FACES_GRID_NUM_CELLS_ROWS,
      Conf.FACES_GRID_NUM_CELLS_COLS,
    )
    ViewsUtils.loadImgFromBase64("FacesGridImg", img)
  }

  /**
   * Get similar faces to encoding from server and load into Faces Grid.
   * @param {[Float]} encoding
   * @param {String} variation Low|Medium|High
   */
  FacesGridView.getSimilarFacesAndLoadIntoGrid = async (
    encoding,
    variation,
  ) => {
    const { img, encodings } = await API.getSimilarFaces(encoding, variation)
    FacesGridModel.saveEncodings(
      encodings,
      Conf.FACES_GRID_NUM_CELLS_ROWS,
      Conf.FACES_GRID_NUM_CELLS_COLS,
    )
    ViewsUtils.loadImgFromBase64("FacesGridImg", img)
    ViewsUtils.scrollTo("FacesGridImg")
  }

  return FacesGridView
})()
