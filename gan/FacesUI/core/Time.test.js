process.env.NODE_ENV = "local"
process.env.TEST = "true"

const async = require("async")
const chai = require("chai")
const should = chai.should()

const Time = require("./Time.js")

describe("Time", () => {

    it("Time.getTimeYYYYMMDDHHMMSSMS", done => {
        const datetime = new Date("2018-05-21T01:49:44.862Z")
        const time = Time.getTimeYYYYMMDDHHMMSSMS(datetime)
        time.should.contain("2018-05-21-01-49-44-862-")
        done()
    })

})