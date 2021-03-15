
## :rocket: 如果有帮助，点个star！:star: ##

### 移动端TNN部署，摄像头实时捕获视频流进行检测。

## iOS:
- Xcode 11.5
- macOS 10.15.4
- iPhone 6sp 13.5.1

## Android:
- Android Studio 4.1.1
- Win10 20H2
- Meizu 16x 8.1.0 (CPU:Qualcomm 710 GPU:Adreno 616)

安卓已经增加权限申请，但如果还是闪退请手动确认下相关权限是否允许。

> Android
```
从界面中选择需要测试的模型。
```
> iOS
```
从界面中选择需要测试的模型。
```

### 模型
| model | android | iOS | from |
|-------------------|:--------:|:--------:|:--------:|
| YOLOv5s           | yes | yes |  [Github](https://github.com/ultralytics/yolov5)   |
| NanoDet           | yes | yes |  [Github](https://github.com/RangiLyu/nanodet)   |

### iOS:
- 如果缺少模型请从 "android_TNN_Demo\app\src\main\assets" 复制 .tnnproto 和 .tnnmodel 文件到 "iOS_TNN_Demo\TNNDemo\res" 下。
- iOS如果opencv2.framework有用到也需要重新下载并替换到工程。
- iOS默认使用的库为scripts/build_ios.sh编译生成。

### Android：
* 由于手机性能、图像尺寸等因素导致FPS在不同手机上相差比较大。该项目主要测试TNN框架的使用，具体模型的转换可以去TNN官方查看转换教程。<br/>
* 由于opencv库太大只保留 arm64-v8a/armeabi-v7a 有需要其它版本的自己去官方下载。
* AS版本不一样可能编译会有各种问题，如果编译错误无法解决、建议使用AS4.0以上版本尝试一下。

由于TNN官方还处于开发阶段，不同时间版本可能会出现功能异常或速度差距比较大都是正常的(当前版本功能正常，但速度变慢了)。

懒人本地转换(不会上传模型): [xxxx -> tnn](https://convertmodel.com/)

轻量级OpenCV:[opencv-mobile](https://github.com/nihui/opencv-mobile)

:art: 截图<br/>

| Android | iOS |
|:-----:|:-----:|
|<img width="324" height="145" src="https://github.com/cmdbug/TNN_Demo/blob/main/Screenshots/Android_CPU_or_GPU.jpg"/>| <img width="320" height="166" src="https://github.com/cmdbug/TNN_Demo/blob/main/Screenshots/iOS_CPU_or_GPU.jpg"/> |

> Android

| YOLOv5s | NanoDet |
|---------|---------|
|<img width="270" height="500" src="https://github.com/cmdbug/TNN_Demo/blob/main/Screenshots/Android_Meizu16x_yolov5s.jpg"/>|<img width="270" height="500" src="https://github.com/cmdbug/TNN_Demo/blob/main/Screenshots/Android_Meizu16x_nanodet.jpg"/>|

> iOS

| YOLOv5s | NanoDet |
|---------|---------|
| <img width="270" height="480" src="https://github.com/cmdbug/TNN_Demo/blob/main/Screenshots/iOS_iPhone6sp_yolov5s_gpu.jpg"/> | <img width="270" height="480" src="https://github.com/cmdbug/TNN_Demo/blob/main/Screenshots/iOS_iPhone5s_nanodet.jpg"/> |


感谢:<br/>
- https://github.com/Tencent/TNN

