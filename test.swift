
import Python
import TensorFlow

let np = Python.import("numpy")
let plt = Python.import("matplotlib.pyplot")


func f(_ x: Float) -> Float {
    return 4 * x * x * x * x - 0.3 * x * x * x * 4 * x  + 1
}

var xs: [Float] = Array(numpy: np.linspace(-2.0, 2.0, 100, dtype: np.float32))!
var ys = xs.map(f)

plt.plot(xs, ys)
plt.show()

struct Variable : Layer {

    var value: Tensor<Float>

    init(_ value: Float) {
        self.value = Tensor(value)
    }

    @differentiable
    func call(_ input: Tensor<Float>) -> Tensor<Float> {
        return value
    }
}

func optimize(
    f: @escaping @differentiable (Float) -> Float, 
    from x: Float, 
    lambda: Float = 0.01, 
    for n: Int = 100
    ) -> [[Float]] {
    
    var sdg_xs = [x]
    var adam_xs = [x]
    var sdg_x = Variable(x)
    var adam_x = Variable(x)

    let sdg = SGD(for: sdg_x, learningRate: lambda)
    let adam = Adam(for: adam_x, learningRate: lambda)

    for _ in 0..<n {

       let sdg_dx = sdg_x.gradient { x -> Float in
            let x = x.value.scalarized()
            return f(x)
       }
        sdg.update(&sdg_x.allDifferentiableVariables, along: sdg_dx)
        sdg_xs.append(sdg_x.value.scalarized())

        let adam_dx = adam_x.gradient { x -> Float in
            let x = x.value.scalarized()
            return f(x)
       }
        adam.update(&adam_x.allDifferentiableVariables, along: adam_dx)
        adam_xs.append(adam_x.value.scalarized())
    }

    return [sdg_xs, adam_xs]
}

let labels = ["sdg", "adam"]
plt.plot(xs, ys, label: "f")


for (xs, label) in zip(optimize(f: f, from: 1.0, lambda: 0.1, for: 1000), labels) {
    ys = xs.map(f)
    plt.plot(xs, ys, label: "\(label)_path")
    plt.scatter(xs.suffix(1), ys.suffix(1), label: label)
}
plt.legend()
plt.show()