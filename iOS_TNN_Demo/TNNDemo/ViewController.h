//
//  ViewController.h
//  TNNDemo
//
//  Created by WZTENG on 2021/1/11.
//  Copyright © 2021 TENG. All rights reserved.
//

#import <UIKit/UIKit.h>

#define W_YOLOV5S 1
#define W_NANODET 2


@interface ViewController : UIViewController

// 1:yolov5s
@property (assign, nonatomic) int USE_MODEL;
@property (assign, nonatomic) bool USE_GPU;


@end
