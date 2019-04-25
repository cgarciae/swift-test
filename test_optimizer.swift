import TensorFlow

struct Variable : Layer {

    var x: Float

    @differentiable
    func call(_ input: Float) -> Float {
        return x 
    }
}

var layer = Variable(x: 1.0)
let optimizer = Adam(for: layer)


let delta = layer.gradient { layer -> Float in
    let x = layer.x
    return x * x
}

print("Layer0:", layer)
print("Delta:", delta)

optimizer.update(&layer.allDifferentiableVariables, along: delta)

print("Layer1:", layer)



    



