import org.jetbrains.kotlinx.multik.api.*
import org.jetbrains.kotlinx.multik.api.linalg.dot
import org.jetbrains.kotlinx.multik.ndarray.data.*
import org.jetbrains.kotlinx.multik.ndarray.operations.*
import kotlin.math.exp

class NeuroLayer(
    inputSignalsCount: Int,
    outputSignalsCount: Int,
) {

    private var inputSignals: D1Array<Double> = mk.zeros(inputSignalsCount)
    private var outputSignals: D1Array<Double> = mk.zeros(outputSignalsCount)
    private var weights = mk.d2arrayIndices(outputSignals.size, inputSignals.size) { _, _ ->
        Math.random()
    }


    fun setInputSignals(inputSignals: List<Double>) {
        if (inputSignals.size == this.inputSignals.size) {
            this.inputSignals = mk.ndarray(inputSignals).asD1Array()
        }
    }

    fun calculateOutputSignals() {
        outputSignals = weights dot inputSignals
    }

    fun getMaxOutputValue() : Pair <Int, Double> {
        val max = outputSignals.max() ?: outputSignals[0]
        return outputSignals.indexOf(max) to max
    }

    fun normalizeOutputSignals() {
        outputSignals = outputSignals / outputSignals.map { exp(it) }.sum()
    }

    fun recalculateWeights(answers: List<Double>) {
        if (answers.size != outputSignals.size) throw Exception("Число выходов не совпадает с числом правильных ответов!");
        val error = outputSignals - mk.ndarray(answers).asD1Array()

        val gradient = error * outputSignals.map { it * (1 - it) }

        for ( j in 0 until weights.shape[0]) {
            for ( i in 0 until  weights.shape[1])
                weights[j, i] -= 0.9 * gradient[j] * inputSignals[i]
        }
    }
}

class NumbersNeuroWeb {
    private val neuroLayer = NeuroLayer(15, 10)

    fun train(answers: List<Double>) = neuroLayer.recalculateWeights(answers)
    fun predict(input: List<Double>): Pair<Int, Double> {
        neuroLayer.setInputSignals(input)
        neuroLayer.calculateOutputSignals()
        neuroLayer.normalizeOutputSignals()
        return neuroLayer.getMaxOutputValue()
    }
}

infix fun NDArray<Double, D1>.multiplyMatrices(other: NDArray <Double, D2>): NDArray<Double, D1> {
    return TODO()
}