import Python
import TensorFlow

let plt = Python.import("matplotlib.pyplot")
let np = Python.import("numpy")

struct Variable : Layer {

    var x: Float

    @differentiable
    func call(_ input: Float) -> Float {
        return x 
    }
}

func optimize(
    at x: Float, 
    lambda: Float = 0.001,
    max_steps: Int = 100,
    f: @escaping @differentiable (Float) -> Float) 
    -> ([Float], [Float], ((Float) -> Float)) {

    var layer = Variable(x: x)
    let optimizer = Adam(for: layer)


    var xs: [Float] = [x]
    var ys: [Float] = [f(x)]

    for _ in 0..<max_steps {

        let dx = layer.gradient { layer -> Float in
            f(layer.x)
        }

        print(dx, layer)
        
        optimizer.update(&layer.allDifferentiableVariables, along: dx)

        print(layer)

        xs.append(layer.x)
        ys.append(f(layer.x))
    }

    return (xs, ys, f)
}

let (xs, ys, f) = optimize(at: 1.0, lambda: 0.05) { x in
    x * x + 1.0
}

var X = Array<Float>(numpy: np.linspace(-5.0, 5.0, 100, dtype: np.float32))!
var Y = X.map(f)

// plt.plot(X, Y)
plt.plot(xs, ys)
plt.show()

