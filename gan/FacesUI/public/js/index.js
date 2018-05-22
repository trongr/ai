const Main = async () => {
    try {
        const { status, img, encodings } = await API.getRandomFaces()
        Views.loadImgFromBase64("RandomFacesImg", img)
    } catch (er) {
        console.error(JSON.stringify(er, null, 4))
    }
}

window.onload = () => Main()