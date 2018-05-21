const randomstring = require("randomstring")
const Time = module.exports = {}

/**
 * Use this to uniquely identify file/folder names based on time and a random
 * string.
 * @param {*} datetime Optionally specify your own time, default is Date.now().
 * @returns {string} time that looks like "2018-05-20-21-27-22-123-XXXXX" where
 * XXXXX is a random string
 */
Time.getTimeYYYYMMDDHHMMSSMS = (datetime = Date.now()) => {
    return new Date(datetime).toISOString().replace(/\D/g, "-") + randomstring.generate(5)
}