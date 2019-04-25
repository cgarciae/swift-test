// Copyright 2018 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import Foundation
import TensorFlow
import Python

let np = Python.import("numpy")
let plt = Python.import("matplotlib.pyplot")

/// Reads a file into an array of bytes.
// func readFile(_ filename: String) -> [UInt8] {
//     let possibleFolders = [".", "MNIST"]
//     for folder in possibleFolders {
//         let parent = URL(fileURLWithPath: folder)
//         let filePath = parent.appendingPathComponent(filename).path
//         guard FileManager.default.fileExists(atPath: filePath) else {
//             continue
//         }
//         let d = Python.open(filePath, "rb").read()
//         return Array(numpy: np.frombuffer(d, dtype: np.uint8))!
//     }
//     fatalError(
//         "Failed to find file with name \(filename) in the following folders: \(possibleFolders).")
// }

func readFile(_ filename: String) -> [UInt8] {
    let d = Python.open(filename, "rb").read()
    return Array(numpy: np.frombuffer(d, dtype: np.uint8))!
}

/// Reads MNIST images and labels from specified file paths.
func readMNIST(imagesFile: String, labelsFile: String) -> (images: Tensor<Float>,
                                                           labels: Tensor<Int32>) {
    print("Reading data.")
    let images = readFile(imagesFile).dropFirst(16).map { Float($0) }
    let labels = readFile(labelsFile).dropFirst(8).map { Int32($0) }
    let rowCount = labels.count
    let imageHeight = 28, imageWidth = 28

    print("Constructing data tensors.")
    return (
        images: Tensor(shape: [rowCount, 1, imageHeight, imageWidth], scalars: images)
            .transposed(withPermutations: [0, 2, 3, 1]) / 255, // NHWC
        labels: Tensor(labels)
    )
}

/// A classifier.
struct GlobalAvgPooling2D : Layer {
    
    @differentiable
    func call(_ input: Tensor<Float>) -> Tensor<Float> {
        // print("endIndex", input.shape.endIndex, input.shape)
        let lastDim = input.shape[input.shape.endIndex - 1]


        return input
            .mean(alongAxes: [1, 2])
            .reshaped(toShape: Tensor<Int32>([-1, Int32(lastDim)]))
    }
}

/// A classifier.
struct Classifier: Layer {

    var conv1 = Conv2D<Float>(filterShape: (3, 3, 1, 32), strides: (2, 2), padding: .same, activation: relu)
    var bn1 = BatchNorm<Float>(featureCount: 32)
    var conv2 = Conv2D<Float>(filterShape: (3, 3, 32, 64), activation: relu)
    var bn2 = BatchNorm<Float>(featureCount: 64)
    // var conv3 = Conv2D<Float>(filterShape: (3, 3, 32, 128), activation: relu)
    // var bn3 = BatchNorm<Float>(featureCount: 128)
    var conv4 = Conv2D<Float>(filterShape: (3, 3, 64, 10))
    var gap = GlobalAvgPooling2D()

    @differentiable
    func call(_ input: Tensor<Float>) -> Tensor<Float> {
        var net = input

        net = conv1(net)
        net = bn1(net)
        net = conv2(net)
        net = bn2(net)
        // net = conv3(net)
        // net = bn3(net)
        net = conv4(net)
        net = gap(net)


        return net
    }
}

let epochCount = 12
let batchSize = 100
let metric_steps = 100

func minibatch<Scalar>(in x: Tensor<Scalar>, at index: Int) -> Tensor<Scalar> {
    let start = index * batchSize
    return x[start..<start+batchSize]
}

let (images, numericLabels) = readMNIST(imagesFile: "data/train-images-idx3-ubyte",
                                        labelsFile: "data/train-labels-idx1-ubyte")
let labels = Tensor<Float>(oneHotAtIndices: numericLabels, depth: 10)

var classifier = Classifier()
let optimizer = RMSProp(for: classifier, learningRate: 0.001)

// metrics stuff
var losses: [Float] = []
var accuracies: [Float] = []
var step = 0

// set phase
Context.local.learningPhase = .training

// The training loop.
for epoch in 1...epochCount {
    var correctGuessCount = 0
    var totalGuessCount = 0
    var totalLoss: Float = 0
    for i in 0 ..< Int(labels.shape[0]) / batchSize {
        step += 1
        let x = minibatch(in: images, at: i)
        let y = minibatch(in: numericLabels, at: i)
        // Compute the gradient with respect to the model.
        let 𝛁model = classifier.gradient { classifier -> Tensor<Float> in
            let ŷ = classifier(x)
            let correctPredictions = ŷ.argmax(squeezingAxis: 1) .== y
            correctGuessCount += Int(Tensor<Int32>(correctPredictions).sum().scalarized())
            totalGuessCount += batchSize
            let loss = softmaxCrossEntropy(logits: ŷ, labels: y)
            totalLoss += loss.scalarized()
            return loss
        }
        // Update the model's differentiable variables along the gradient vector.
        optimizer.update(&classifier.allDifferentiableVariables, along: 𝛁model)

        // metrics
        if step % metric_steps == 0 {
            let accuracy = Float(correctGuessCount) / Float(totalGuessCount)

            print("""
                [Epoch \(epoch)] \
                Loss: \(totalLoss), \
                Accuracy: \(correctGuessCount)/\(totalGuessCount) (\(accuracy))
                """)

            losses.append(totalLoss / Float(metric_steps))
            accuracies.append(accuracy)

            plt.clf()
            
            plt.subplot(1, 2, 1)
            plt.plot(losses)
            plt.title("Loss")
            
            
            plt.subplot(1, 2, 2)
            plt.plot(accuracies)
            plt.title("Accuracy")

            plt.draw()
            plt.pause(0.1)

            // reset metrics
            correctGuessCount = 0
            totalGuessCount = 0
            totalLoss = 0
        }
    }
}