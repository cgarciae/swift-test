
import Python
import TensorFlow

let np = Python.import("numpy")
let plt = Python.import("matplotlib.pyplot")


func f(_ x: Float) -> Float {
    return x * x + 1
}

var xs: [Float] = Array(numpy: np.linspace(-2.0, 2.0, 100, dtype: np.float32))!
var ys = xs.map(f)

plt.plot(xs, ys)
plt.show()
plt.plot(xs, ys)

func optimize(
    f: @escaping @differentiable (Float) -> Float, 
    from x: Float, lambda: Float = 0.01, 
    for n: Int = 100) -> [Float] {
    
    var xs = [x]
    var x = x

    let df = gradient(of: f)

    for _ in 0..<n {

        x -= lambda * df(x)

        xs.append(x)
    }

    return xs
}


xs = optimize(f: f, from: 1.0, for: 200)
ys = xs.map(f)

plt.plot(xs, ys)
plt.show()