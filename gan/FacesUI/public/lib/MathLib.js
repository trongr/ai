const MathLib = (() => {
    const MathLib = {}

    /**
     * Wrap x around the interval [-1, 1). Essentially:
     * f(x) => {
     *      if (x >= -1 && x < 1) return x
     *      else if (x > 1) return x % 1 - 1
     *      else return x % 1 + 1
     * }
     * @param {Float} x Value to wrap around [-1, 1)
     */
    MathLib.WrapXAroundMinusOneAndOne = (x) => {
        return 2 * (((((x + 1) / 2) % 1) + 1) % 1) - 1
    }

    return MathLib
})();