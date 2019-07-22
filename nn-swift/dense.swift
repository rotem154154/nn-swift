//
//  dense.swift
//  nn_diy
//
//  Created by rotem israeli on 21/07/2019.
//  Copyright © 2019 TandR. All rights reserved.
//

import Foundation
import Accelerate


class dense{
  var input_size : Int
  var hidden_size : Int
  var output_size : Int
  var weights1 : [Float32]
  var weights2 : [Float32]
  var input_descriptor : BNNSVectorDescriptor
  var hidden_descriptor : BNNSVectorDescriptor
  var output_descriptor : BNNSVectorDescriptor
  var filter1 : BNNSFilter
  var filter2 : BNNSFilter
  var filter1LayerParameters = BNNSFullyConnectedLayerParameters()
  var filter2LayerParameters = BNNSFullyConnectedLayerParameters()
  var input_stack : [Float32]
  var hidden_stack : [Float32]
  var output_stack : [Float32]
  
  init(input_size : Int, hidden_size : Int, output_size : Int) {
    self.input_size = input_size
    self.hidden_size = hidden_size
    self.output_size = output_size
    
    self.input_descriptor = BNNSVectorDescriptor(size: input_size,data_type: BNNSDataType.float,data_scale: 0,data_bias: 0)
    self.hidden_descriptor = BNNSVectorDescriptor(size: hidden_size,data_type: BNNSDataType.float,data_scale: 0,data_bias: 0)
    self.output_descriptor = BNNSVectorDescriptor(size: output_size,data_type: BNNSDataType.float,data_scale: 0,data_bias: 0)
    
    weights1 = []
    for _ in 0...input_size*hidden_size{
      weights1.append(Float32.random(in: -1...1))
    }
    weights2 = []
    for _ in 0...hidden_size*output_size{
      weights2.append(Float32.random(in: -1...1))
    }
    
    filter1LayerParameters.in_size = input_size
    filter1LayerParameters.out_size = hidden_size
    filter1LayerParameters.activation = BNNSActivation(function: BNNSActivationFunction.rectifiedLinear)
    filter1LayerParameters.weights = BNNSLayerData(data: weights1, data_type: BNNSDataType.float)
    filter1 = BNNSFilterCreateFullyConnectedLayer(&input_descriptor, &hidden_descriptor, &filter1LayerParameters, nil)!
    filter2LayerParameters.in_size = hidden_size
    filter2LayerParameters.out_size = output_size
    filter2LayerParameters.activation = BNNSActivation(function: BNNSActivationFunction.softmax)
    filter2LayerParameters.weights = BNNSLayerData(data: weights2, data_type: BNNSDataType.float)
    filter2 = BNNSFilterCreateFullyConnectedLayer(&hidden_descriptor, &output_descriptor, &filter2LayerParameters, nil)!
    print(output_size)
    input_stack = Array(repeating: 0, count: input_size)
    hidden_stack = Array(repeating: 0, count: hidden_size)
    output_stack = Array(repeating: 0, count: output_size)
  }
  
  func forward(input_stack : [Float32]) -> [Float32] {
    self.input_stack = input_stack
    BNNSFilterApply(filter1, &self.input_stack, &self.hidden_stack)
    BNNSFilterApply(filter2, &self.hidden_stack, &self.output_stack)
    
    
    return output_stack
  }
  
  
}




















//
//
//
//var inputDescriptor = BNNSVectorDescriptor(size: 3,
//                                           data_type: BNNSDataType.float,
//                                           data_scale: 0,
//                                           data_bias: 0)
//var hiddenDescriptor = BNNSVectorDescriptor(size: 4,
//                                            data_type: BNNSDataType.float,
//                                            data_scale: 0,
//                                            data_bias: 0)
//var outputDescriptor = BNNSVectorDescriptor(size: 1,
//                                            data_type: BNNSDataType.float,
//                                            data_scale: 0,
//                                            data_bias: 0)
//
//
//// Trained weights
//let filter1Weights: [Float32] = [4.6013571, -2.58413484, 0.97538679,
//                                 4.17197193, -5.81447929, -2.02685775,
//                                 -6.30956245, -6.60793435, 2.52949751,
//                                 -4.19745118, -3.68396123, 5.84371739]
//
//let filter2Weights: [Float32] = [-6.96765763, 7.14101949, -10.31917382, 7.86128405]
//
//
//// Input to hidden layer.
//var filter1LayerParameters = BNNSFullyConnectedLayerParameters()
//
//filter1LayerParameters.in_size = inputDescriptor.size
//filter1LayerParameters.out_size = hiddenDescriptor.size
//filter1LayerParameters.activation = BNNSActivation(function: BNNSActivationFunction.sigmoid,
//                                                   alpha: 0,
//                                                   beta: 0)
//filter1LayerParameters.weights = BNNSLayerData(data: filter1Weights,
//                                               data_type: BNNSDataType.float,
//                                               data_scale: 0,
//                                               data_bias: 0,
//                                               data_table: nil)
//
//let filter1 = BNNSFilterCreateFullyConnectedLayer(&inputDescriptor, &hiddenDescriptor, &filter1LayerParameters, nil)
//
//// Hidden to output layer.
//var filter2LayerParameters = BNNSFullyConnectedLayerParameters()
//
//filter2LayerParameters.in_size = hiddenDescriptor.size
//filter2LayerParameters.out_size = outputDescriptor.size
//filter2LayerParameters.activation = BNNSActivation(function: BNNSActivationFunction.sigmoid,
//                                                   alpha: 0,
//                                                   beta: 0)
//filter2LayerParameters.weights = BNNSLayerData(data: filter2Weights,
//                                               data_type: BNNSDataType.float,
//                                               data_scale: 0,
//                                               data_bias: 0,
//                                               data_table: nil)
//
//let filter2 = BNNSFilterCreateFullyConnectedLayer(&hiddenDescriptor, &outputDescriptor, &filter2LayerParameters, nil)
//
//
//
//var inputStack: [Float32] = [0, 0, 1,
//                             1, 1, 1,
//                             0, 0, 0,
//                             0, 1, 0]
//
//var hiddenStack: [Float32] = [0, 0, 0, 0,
//                              0, 0, 0, 0,
//                              0, 0, 0, 0,
//                              0, 0, 0, 0]
//
//var outputStack: [Float32] = [0, 0, 0, 0]
//
//BNNSFilterApplyBatch(filter1, 4, &inputStack, inputDescriptor.size, &hiddenStack, hiddenDescriptor.size)
//BNNSFilterApplyBatch(filter2, 4, &hiddenStack, hiddenDescriptor.size, &outputStack, outputDescriptor.size)
//
//outputStack.forEach { (output) in
//  print(output)
//}
