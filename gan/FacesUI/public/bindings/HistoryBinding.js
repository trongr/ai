const HistoryBinding = (() => {
    const HistoryBinding = {}

    HistoryBinding.init = () => {
        bindHistoryImgClick()
    }

    function bindHistoryImgClick() {
        const tag = "HistoryBinding.bindHistoryImgClick"
        $('body').on('click', "#HistoryBox .HistoryImg", function (e) {
            const idx = $(this).data("historyimgidx")
            const encoding = HistoryModel.getEncoding(idx)
            CurrentFaceView.getFaceByEncodingAndLoadIntoCurrentFace(encoding)
        })
    }

    return HistoryBinding
})();