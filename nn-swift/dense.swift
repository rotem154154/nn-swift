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
  var num_layers : Int
  var layers_sizes : [Int] = []
  var weights : [[Float32]] = []
  var stacks : [[Float32]] = []
  var layers_descriptor : [BNNSVectorDescriptor] = []
  var filters : [BNNSFilter] = []
  var filters_parameters : [BNNSFullyConnectedLayerParameters] = []
  
  init(layers_sizes : [Int]) {
    self.num_layers = layers_sizes.count
    self.layers_sizes = layers_sizes
    
//    self.layers_descriptor = []
//    self.stacks = []
    for i in 0..<num_layers{
      self.layers_descriptor.append(BNNSVectorDescriptor(size: layers_sizes[i], data_type: BNNSDataType.float))
      self.stacks.append(Array(repeating: Float32(0), count: layers_sizes[i]))
    }
    
//    self.weights = []
    for i in 0..<num_layers-1{
      var w : [Float32] = []
      for _ in 0..<layers_sizes[i]*layers_sizes[i+1]{
        w.append(Float32(drand48()))
      }
      weights.append(w)
      
    }
    
//    self.filters_parameters = []
//    self.filters = []
    for i in 0..<num_layers-1{
      var fp = BNNSFullyConnectedLayerParameters()
      fp.in_size = layers_sizes[i]
      fp.out_size = layers_sizes[i+1]
      fp.activation = BNNSActivation(function: BNNSActivationFunction.rectifiedLinear)
      if i == num_layers-2{
        fp.activation = BNNSActivation(function: BNNSActivationFunction.softmax)
      }
      fp.weights = BNNSLayerData(data: weights[i], data_type: BNNSDataType.float)
      filters_parameters.append(fp)
      filters.append(BNNSFilterCreateFullyConnectedLayer(&self.layers_descriptor[i], &self.layers_descriptor[i+1], &filters_parameters[i], nil)!)
    }
    
  }
  
  func forward(input_stack : [Float32]) -> [Float32] {
    self.stacks[0] = input_stack
//    for i in 0..<num_layers-1{
////      print(self.stacks[i])
//      BNNSFilterApply(filters[i], &self.stacks[i], &self.stacks[i+1])
//
//    }
    switch num_layers {
    case 2:
      BNNSFilterApply(filters[0], &self.stacks[0], &self.stacks[1])
    case 3:
      BNNSFilterApply(filters[0], &self.stacks[0], &self.stacks[1])
      BNNSFilterApply(filters[1], &self.stacks[1], &self.stacks[2])
    case 4:
      BNNSFilterApply(filters[0], &self.stacks[0], &self.stacks[1])
      BNNSFilterApply(filters[1], &self.stacks[1], &self.stacks[2])
      BNNSFilterApply(filters[2], &self.stacks[2], &self.stacks[3])
    case 5:
      BNNSFilterApply(filters[0], &self.stacks[0], &self.stacks[1])
      BNNSFilterApply(filters[1], &self.stacks[1], &self.stacks[2])
      BNNSFilterApply(filters[2], &self.stacks[2], &self.stacks[3])
      BNNSFilterApply(filters[3], &self.stacks[3], &self.stacks[4])
    case 6:
      BNNSFilterApply(filters[0], &self.stacks[0], &self.stacks[1])
      BNNSFilterApply(filters[1], &self.stacks[1], &self.stacks[2])
      BNNSFilterApply(filters[2], &self.stacks[2], &self.stacks[3])
      BNNSFilterApply(filters[3], &self.stacks[3], &self.stacks[4])
      BNNSFilterApply(filters[4], &self.stacks[4], &self.stacks[5])
    case 7:
      BNNSFilterApply(filters[0], &self.stacks[0], &self.stacks[1])
      BNNSFilterApply(filters[1], &self.stacks[1], &self.stacks[2])
      BNNSFilterApply(filters[2], &self.stacks[2], &self.stacks[3])
      BNNSFilterApply(filters[3], &self.stacks[3], &self.stacks[4])
      BNNSFilterApply(filters[4], &self.stacks[4], &self.stacks[5])
      BNNSFilterApply(filters[5], &self.stacks[5], &self.stacks[6])
    case 8:
      BNNSFilterApply(filters[0], &self.stacks[0], &self.stacks[1])
      BNNSFilterApply(filters[1], &self.stacks[1], &self.stacks[2])
      BNNSFilterApply(filters[2], &self.stacks[2], &self.stacks[3])
      BNNSFilterApply(filters[3], &self.stacks[3], &self.stacks[4])
      BNNSFilterApply(filters[4], &self.stacks[4], &self.stacks[5])
      BNNSFilterApply(filters[5], &self.stacks[5], &self.stacks[6])
      BNNSFilterApply(filters[6], &self.stacks[6], &self.stacks[7])
    default:
      print("too much layers")
    }
    return stacks.last!
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
