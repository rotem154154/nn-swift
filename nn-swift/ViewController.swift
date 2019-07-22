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
    
    
    let model = dense(layers_sizes: [3,4,4,3,7])
    
    let input = Array(repeating: Float32(0.0), count: model.layers_sizes[0])
    
    let out = model.forward(input_stack: input)
    
    print(out)
    
    
    
    
    
    
  }
  
  
}

