const API = (() => {
    const API = {}

    API.getRandomFaces = () => $.ajax({ url: "/faces", type: "GET", data: {} })
    console.log("")
    return API
})();