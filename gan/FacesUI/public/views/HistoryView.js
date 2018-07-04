const HistoryView = (() => {
    const HistoryView = {}

    /**
     * Save the encoding in HistoryModel and load the img into HistoryBox
     * @param {*} encoding
     * @param {*} img
     */
    HistoryView.saveEncodingImg = (encoding, img) => {
        const idx = HistoryModel.count()
        HistoryModel.push(encoding)
        HistoryView.loadImgIntoHistoryBox(idx, img)
    }

    /**
     *
     * @param {*} idx The index of the encoding in the history model, to be set
     * as the img key in HistoryBox
     * @param {*} img
     */
    HistoryView.loadImgIntoHistoryBox = (idx, img) => {
        const HistoryImgBox = $(
            `<div class="HistoryBoxItem">
                <div class="HistoryEncodingBox">

                </div>
                <img id="HistoryImg${idx}"
                    class="HistoryImg" data-historyimgidx="${idx}"
                    src="${img}">
             </div>`
        )
        HistoryImgBox.prependTo("#HistoryBox")
    }

    return HistoryView
})();