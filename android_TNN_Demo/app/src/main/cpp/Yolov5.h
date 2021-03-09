#ifndef YOLOV5S_H
#define YOLOV5S_H

#include <android/log.h>
#include <string>
#include <vector>
#include <jni.h>

#include "tnn/core/tnn.h"
#include "tnn/core/blob.h"
#include "tnn/utils/blob_converter.h"
#include "tnn/utils/mat_utils.h"

#ifndef LOG_TAG
#define LOG_TAG "WZT_TNN"
#define TLOGD(...) __android_log_print(ANDROID_LOG_DEBUG,LOG_TAG ,__VA_ARGS__) // 定义LOGD类型
#define TLOGI(...) __android_log_print(ANDROID_LOG_INFO,LOG_TAG ,__VA_ARGS__) // 定义LOGI类型
#define TLOGW(...) __android_log_print(ANDROID_LOG_WARN,LOG_TAG ,__VA_ARGS__) // 定义LOGW类型
#define TLOGE(...) __android_log_print(ANDROID_LOG_ERROR,LOG_TAG ,__VA_ARGS__) // 定义LOGE类型
#define TLOGF(...) __android_log_print(ANDROID_LOG_FATAL,LOG_TAG ,__VA_ARGS__) // 定义LOGF类型
#endif


namespace yolocv {
    typedef struct {
        int width;
        int height;
    } YoloSize;
}

typedef struct {
    std::string name;
    int stride;
    std::vector<yolocv::YoloSize> anchors;
} YoloLayerData;

typedef struct BoxInfo {
    float x1;
    float y1;
    float x2;
    float y2;
    float score;
    int label;
} BoxInfo;

char *jstring2string(JNIEnv *env, jstring jstr);
std::string fdLoadFile(std::string path);

class YoloV5 {
public:
    YoloV5(std::string proto, std::string model, bool useGPU);

    ~YoloV5();

    std::vector<BoxInfo> detect(JNIEnv *env, jobject bitmap, float threshold, float nms_threshold);

    void generateDetectResult(std::vector<std::shared_ptr<TNN_NS::Mat>> outputs, std::vector<BoxInfo> &detecs,
                              float threshold, float nms_threshold);

    void nms(std::vector<BoxInfo> &input_boxes, float nms_thresh);

    std::vector<std::string> labels{"person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
                                    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
                                    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
                                    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
                                    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
                                    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
                                    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
                                    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
                                    "hair drier", "toothbrush"};
private:
    std::shared_ptr<tnn::TNN> net;
    std::shared_ptr<TNN_NS::Instance> instance;
    TNN_NS::DeviceType device_type;

    int net_width = 640;
    int net_height = 448;
    int num_class = 80;

    std::vector<YoloLayerData> layers{
            {"output", 32, {{116, 90}, {156, 198}, {373, 326}}},
            {"413", 16, {{30,  61}, {62,  45},  {59,  119}}},
            {"431", 8,  {{10,  13}, {16,  30},  {33,  23}}},
    };

public:
    static YoloV5 *detector;
    static bool hasGPU;
    static bool toUseGPU;
};

#endif //YOLOV5S_H
