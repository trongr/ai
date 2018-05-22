process.env.NODE_ENV = "local"
process.env.TEST = "true"

const async = require("async")
const chai = require("chai")
const should = chai.should()

const FS = require("./FS.js")

describe("FS", () => {

    it("FS.readTxtFile", done => {
        const filepath = "./app.js"
        FS.readTxtFile(filepath, (er, data) => {
            // console.log(data, er)
            should.not.exist(er)
            data.should.contain("var app")
            done()
        })
    })

})