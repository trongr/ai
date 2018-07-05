const HistoryModel = (() => {
    const HistoryModel = {}

    // List of encodings loaded previously into the Current Face. Oldest first.
    const encodings = []

    HistoryModel.push = (encoding) => {
        encodings.push(_.cloneDeep(encoding))
    }

    /**
     * Return the encoding at index idx
     * @param {*} idx
     */
    HistoryModel.getEncoding = (idx) => {
        return _.cloneDeep(encodings[idx])
    }

    HistoryModel.getAllEncodings = () => {
        return _.cloneDeep(encodings)
    }

    /**
     * Each encoding in encodings corresponds by index to the img in HistoryBox,
     * so e.g. can use this count to get the index and set it as the img ID.
     */
    HistoryModel.count = () => {
        return encodings.length
    }

    return HistoryModel
})();