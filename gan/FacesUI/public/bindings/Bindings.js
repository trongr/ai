const Bindings = (() => {
    const Bindings = {}

    Bindings.init = () => {
        FacesGridBinding.init()
        CurrentFaceBinding.init()
        HistoryBinding.init()
    }

    return Bindings
})();