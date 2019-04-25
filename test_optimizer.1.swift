import TensorFlow

struct Variable : Layer {

    var x: Tensor<Float>

    @differentiable
    func call(_ input: Tensor<Float>) -> Tensor<Float> {
        return x 
    }
}


var layer = Variable(x: Tensor(1.0))
let optimizer = Adam(for: layer)


let delta = layer.gradient { layer -> Tensor<Float> in
    let x = layer.x
    return x * x
}

print("Layer0:", layer)
print("Delta:", delta)

optimizer.update(&layer.allDifferentiableVariables, along: delta)

print("Layer1:", layer)

    



