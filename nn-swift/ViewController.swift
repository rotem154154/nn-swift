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
    
    
    let model = dense(layers_sizes: [3,4,5,3])
    
    let input : [Float32] = [0,1,1]
    
    model.random_weights()
    
    model.mutate(alpha: 0.01)
    
    print(model.forward(input_stack: input))
    
    model.save_weights(key: "weights1")
    model.load_weights(key: "weights1")
    
  }
  
  
}

