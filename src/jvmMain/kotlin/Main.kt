import androidx.compose.desktop.ui.tooling.preview.Preview
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.material.Button
import androidx.compose.material.MaterialTheme
import androidx.compose.material.Text
import androidx.compose.material.TextField
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.compose.ui.window.Window
import androidx.compose.ui.window.application
import org.jetbrains.kotlinx.multik.api.d1array
import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.ndarray.operations.toList


val numbers = mapOf(
    0 to mk[
        mk[1.0, 1.0, 1.0],
        mk[1.0, 0.0, 1.0],
        mk[1.0, 0.0, 1.0],
        mk[1.0, 0.0, 1.0],
        mk[1.0, 1.0, 1.0],
    ],
    1 to mk[
        mk[0.0, 0.0, 1.0],
        mk[0.0, 0.0, 1.0],
        mk[0.0, 0.0, 1.0],
        mk[0.0, 0.0, 1.0],
        mk[0.0, 0.0, 1.0],
    ],
    2 to mk[
        mk[1.0, 1.0, 1.0],
        mk[0.0, 0.0, 1.0],
        mk[1.0, 1.0, 1.0],
        mk[1.0, 0.0, 1.0],
        mk[1.0, 1.0, 1.0],
    ],
    3 to mk[
        mk[1.0, 1.0, 1.0],
        mk[0.0, 0.0, 1.0],
        mk[1.0, 1.0, 1.0],
        mk[0.0, 0.0, 1.0],
        mk[1.0, 1.0, 1.0],
    ],
    4 to mk[
        mk[1.0, 0.0, 1.0],
        mk[1.0, 0.0, 1.0],
        mk[1.0, 1.0, 1.0],
        mk[0.0, 0.0, 1.0],
        mk[0.0, 0.0, 1.0],
    ],
    5 to mk[
        mk[1.0, 1.0, 1.0],
        mk[1.0, 0.0, 0.0],
        mk[1.0, 1.0, 1.0],
        mk[0.0, 0.0, 1.0],
        mk[1.0, 1.0, 1.0],
    ],
    6 to mk[
        mk[1.0, 1.0, 1.0],
        mk[1.0, 0.0, 0.0],
        mk[1.0, 1.0, 1.0],
        mk[1.0, 0.0, 1.0],
        mk[1.0, 1.0, 1.0],
    ],
    7 to mk[
        mk[1.0, 1.0, 1.0],
        mk[0.0, 0.0, 1.0],
        mk[0.0, 0.0, 1.0],
        mk[0.0, 0.0, 1.0],
        mk[0.0, 0.0, 1.0],
    ],
    8 to mk[
        mk[1.0, 1.0, 1.0],
        mk[1.0, 0.0, 1.0],
        mk[1.0, 1.0, 1.0],
        mk[1.0, 0.0, 1.0],
        mk[1.0, 1.0, 1.0],
    ],
    9 to mk[
        mk[1.0, 1.0, 1.0],
        mk[1.0, 0.0, 1.0],
        mk[1.0, 1.0, 1.0],
        mk[0.0, 0.0, 1.0],
        mk[1.0, 1.0, 1.0],
    ],
)

val train = (0..9).map { it to emptyList<Int>() }.toMap().toMutableMap()

@Composable
@Preview
fun App() {
    val neuroWeb = remember { NumbersNeuroWeb() }
    var selected by remember { mutableStateOf(0) }
    var neuronNumber by remember { mutableStateOf(0) }
    var accurancy by remember { mutableStateOf(0.0) }

    LaunchedEffect(true) {
        repeat(1000) {
            train.keys.forEach {
                selected = it
                val (n, a) = neuroWeb.predict(numbers[selected]!!.flatten())
                neuronNumber = n
                accurancy = a
                var count = 0
                while (neuronNumber != selected) {
                    neuroWeb.train(mk.d1array(10) { if (it == selected) 1.0 else 0.0 }.toList())
                    val (n, a) = neuroWeb.predict(numbers[selected]!!.flatten())
                    neuronNumber = n
                    accurancy = a
                    count++
                }
                if (count != 0) {
                    train[it] = train[it]!! + count
                }
            }
        }
        println(train.map { "${it.key} = ${it.value.size}" })
    }

    MaterialTheme {
        Column(modifier = Modifier.fillMaxSize().padding(16.dp)) {
            Text(neuronNumber.toString())
            Text(accurancy.toString())
            TextField(
                selected.toString(),
                onValueChange = {
                    it.toIntOrNull()?.let { selected = it }
                }
            )
            Button(
                onClick = {
                    val (n, a) = neuroWeb.predict(numbers[selected]!!.flatten())
                    neuronNumber = n
                    accurancy = a
                    var count = 0
                    while (neuronNumber != selected) {
                        neuroWeb.train(mk.d1array(10) { if (it == selected) 1.0 else 0.0 }.toList())
                        val (n, a) = neuroWeb.predict(numbers[selected]!!.flatten())
                        neuronNumber = n
                        accurancy = a
                        count++
                    }
                    if (count != 0) { println("$selected : $count") }
                }
            ) {
                Text("Predict")
            }
        }
    }
}

fun main() = application {
    /*NeuroLayer(
        inputSignals = mk.ndarray(mk[1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]),
        outputSignals = mk.zeros(15)
    )*/
    Window(onCloseRequest = ::exitApplication) {
        App()
    }
}
