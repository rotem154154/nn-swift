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
