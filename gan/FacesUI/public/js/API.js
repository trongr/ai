const API = (() => {
    const API = {}

    API.getRandomFaces = () => $.ajax({ url: "/faces", type: "GET", data: {} })

    return API
})();