const Validate = (module.exports = {})

/**
 * Converts list of strings containing floats into list of floats
 * @param {Array} encoding
 * @returns new list of floats
 */
Validate.sanitizeEncoding = (encoding) => {
  return encoding.map((e) => {
    return parseFloat(e)
  })
}
