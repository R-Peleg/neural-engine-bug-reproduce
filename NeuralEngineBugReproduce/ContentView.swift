//
//  ContentView.swift
//  NeuralEngineBugReproduce
//
//  Created by Reuven Aviad Peleg on 28/11/2022.
//

import SwiftUI
import CoreML

struct ContentView: View {
    var body: some View {
        VStack {
            Text("Neural Engine Reproduction")
            Button("Click here")
            {
                // Init 3 version of the model, for CPU, GPU and NeuralEngine
                let configCpu = MLModelConfiguration()
                configCpu.computeUnits = MLComputeUnits.cpuOnly
                let modelCpu = try! Block(configuration: configCpu)

                let configGpu = MLModelConfiguration()
                configGpu.computeUnits = MLComputeUnits.cpuAndGPU
                let modelGpu = try! Block(configuration: configGpu)
                
                let configNeuralEngine = MLModelConfiguration()
                configNeuralEngine.computeUnits = MLComputeUnits.cpuAndNeuralEngine
                let modelNeuralEngine = try! Block(configuration: configNeuralEngine)
                
                // Prepare random inputs
                let input1 = try! MLMultiArray(
                    shape: [1, 800, 43, 1],
                    dataType:MLMultiArrayDataType.float32)
                for i in 0..<input1.count {
                    input1[i] = Float.random(in: -1...1) as NSNumber
                }
                let input2 = try! MLMultiArray(
                    shape: [1, 128, 43, 1],
                    dataType:MLMultiArrayDataType.float32)
                for i in 0..<input2.count {
                    input2[i] = Float.random(in: -1...1) as NSNumber
                }

                let model_input = BlockInput(input2: input2, input1: input1)

                // Predict
                let predictionCpu = try! modelCpu.prediction(input: model_input)
                let predictionGpu = try! modelGpu.prediction(input: model_input)
                let predictionNeuralEngine = try! modelNeuralEngine.prediction(input: model_input)

                // Inspect results
                var absDiffSum = 0 as Float
                var diffSum = 0 as Float
                var maxAbsDiff = 0 as Float
                var maxAbsDiffIdx = -1
                for i in 0..<predictionCpu.output.count {
                    let diff = predictionNeuralEngine.output[i].floatValue - predictionCpu.output[i].floatValue
                    diffSum += diff
                    absDiffSum += abs(diff)
                    if (abs(diff) > maxAbsDiff) {
                        maxAbsDiff = abs(diff)
                        maxAbsDiffIdx = i
                    }
                }
                print("Average abs diff: ", absDiffSum / Float(predictionCpu.output.count))
                print("Average diff: ", diffSum / Float(predictionCpu.output.count))
                print("Max abs diff ", maxAbsDiff, " at ", maxAbsDiffIdx)
                print("Index\t\tCPU\t\t\tGPU\t\t\tANE")
                for i in maxAbsDiffIdx-5...maxAbsDiffIdx+5 {
                    print(i, "\t",
                          predictionCpu.output[i], "\t",
                          predictionGpu.output[i], "\t",
                          predictionNeuralEngine.output[i])
                }
            }
        }
        .padding()
    }
}

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}
