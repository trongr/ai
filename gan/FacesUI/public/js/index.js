const Main = async () => {
    try {
        const re = await API.getRandomFaces()
        console.log(JSON.stringify(re, null, 4))
    } catch (er) {
        console.error(JSON.stringify(er, null, 4))
    }
}

window.onload = () => Main()