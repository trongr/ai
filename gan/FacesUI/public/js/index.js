

const Main = async () => {
    try {
        const re = await API.getRandomFaces()
        console.log(JSON.stringify(re, null, 4))
    } catch (e) {
        console.error(JSON.stringify(e, null, 4))
    }
}

window.onload = function () {
    Main()
}