const Main = async () => {
    Views.init()
    Bindings.init()
    Models.init()
}

window.onload = () => Main()