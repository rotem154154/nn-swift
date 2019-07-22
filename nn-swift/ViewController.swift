//
//  ViewController.swift
//  nn-swift
//
//  Created by rotem israeli on 22/07/2019.
//  Copyright © 2019 TandR. All rights reserved.
//


import UIKit
import Accelerate

class ViewController: UIViewController {
  
  override func viewDidLoad() {
    super.viewDidLoad()
    
    
    var model = dense(input_size: 3, hidden_size: 4, output_size: 4)
    var out = model.forward(input_stack: [-0.1,-0.51,0])
    print(out)
    
    var inputDescriptor = BNNSVectorDescriptor(size: 3,
                                               data_type: BNNSDataType.float,
                                               data_scale: 0,
                                               data_bias: 0)
    var hiddenDescriptor = BNNSVectorDescriptor(size: 4,
                                                data_type: BNNSDataType.float,
                                                data_scale: 0,
                                                data_bias: 0)
    var outputDescriptor = BNNSVectorDescriptor(size: 1,
                                                data_type: BNNSDataType.float,
                                                data_scale: 0,
                                                data_bias: 0)
    
    
    // Trained weights
    let filter1Weights: [Float32] = [4.6013571, -2.58413484, 0.97538679,
                                     4.17197193, -5.81447929, -2.02685775,
                                     -6.30956245, -6.60793435, 2.52949751,
                                     -4.19745118, -3.68396123, 5.84371739]
    
    let filter2Weights: [Float32] = [-6.96765763, 7.14101949, -10.31917382, 7.86128405]
    
    
    // Input to hidden layer.
    var filter1LayerParameters = BNNSFullyConnectedLayerParameters()
    
    filter1LayerParameters.in_size = inputDescriptor.size
    filter1LayerParameters.out_size = hiddenDescriptor.size
    filter1LayerParameters.activation = BNNSActivation(function: BNNSActivationFunction.sigmoid,
                                                       alpha: 0,
                                                       beta: 0)
    filter1LayerParameters.weights = BNNSLayerData(data: filter1Weights,
                                                   data_type: BNNSDataType.float,
                                                   data_scale: 0,
                                                   data_bias: 0,
                                                   data_table: nil)
    
    let filter1 = BNNSFilterCreateFullyConnectedLayer(&inputDescriptor, &hiddenDescriptor, &filter1LayerParameters, nil)
    
    // Hidden to output layer.
    var filter2LayerParameters = BNNSFullyConnectedLayerParameters()
    
    filter2LayerParameters.in_size = hiddenDescriptor.size
    filter2LayerParameters.out_size = outputDescriptor.size
    filter2LayerParameters.activation = BNNSActivation(function: BNNSActivationFunction.sigmoid,
                                                       alpha: 0,
                                                       beta: 0)
    filter2LayerParameters.weights = BNNSLayerData(data: filter2Weights,
                                                   data_type: BNNSDataType.float,
                                                   data_scale: 0,
                                                   data_bias: 0,
                                                   data_table: nil)
    
    let filter2 = BNNSFilterCreateFullyConnectedLayer(&hiddenDescriptor, &outputDescriptor, &filter2LayerParameters, nil)
    
    
    
    var inputStack: [Float32] = [0, 0, 1,
                                 1, 1, 1,
                                 0, 0, 0,
                                 0, 1, 0]
    
    var hiddenStack: [Float32] = [0, 0, 0, 0,
                                  0, 0, 0, 0,
                                  0, 0, 0, 0,
                                  0, 0, 0, 0]
    
    var outputStack: [Float32] = [0, 0, 0, 0]
    
    BNNSFilterApplyBatch(filter1, 4, &inputStack, inputDescriptor.size, &hiddenStack, hiddenDescriptor.size)
    BNNSFilterApplyBatch(filter2, 4, &hiddenStack, hiddenDescriptor.size, &outputStack, outputDescriptor.size)
    
//    outputStack.forEach { (output) in
//      print(output)
//    }
    
    
    
    
    
    
    
    
    
  }
  
  
}

