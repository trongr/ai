const HistoryModel = (() => {
    const HistoryModel = {}

    // List of encodings loaded previously into the Current Face. Oldest first.
    const encodings = []

    HistoryModel.push = (encoding) => {
        encodings.push(encoding)
    }

    return HistoryModel
})();