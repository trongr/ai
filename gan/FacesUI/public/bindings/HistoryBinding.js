const HistoryBinding = (() => {
    const HistoryBinding = {}

    HistoryBinding.init = () => {
        bindHistoryImgClick()
    }

    function bindHistoryImgClick() {
        const tag = "HistoryBinding.bindHistoryImgClick"
        $('body').on('click', "#HistoryBox .HistoryImg", function (e) {
            console.log("poij you're clicking me!")
            console.log($(this).data("historyimgidx"))
        })
    }

    return HistoryBinding
})();