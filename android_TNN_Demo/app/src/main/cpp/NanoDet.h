#ifndef NANODET_H
#define NANODET_H

#include <android/log.h>
#include <string>
#include <vector>
#include <jni.h>

#include "tnn/core/tnn.h"
#include "tnn/core/blob.h"
#include "tnn/utils/blob_converter.h"
#include "tnn/utils/mat_utils.h"

#include "Yolov5.h"

#ifndef LOG_TAG
#define LOG_TAG "WZT_TNN"
#define TLOGD(...) __android_log_print(ANDROID_LOG_DEBUG,LOG_TAG ,__VA_ARGS__) // 定义LOGD类型
#define TLOGI(...) __android_log_print(ANDROID_LOG_INFO,LOG_TAG ,__VA_ARGS__) // 定义LOGI类型
#define TLOGW(...) __android_log_print(ANDROID_LOG_WARN,LOG_TAG ,__VA_ARGS__) // 定义LOGW类型
#define TLOGE(...) __android_log_print(ANDROID_LOG_ERROR,LOG_TAG ,__VA_ARGS__) // 定义LOGE类型
#define TLOGF(...) __android_log_print(ANDROID_LOG_FATAL,LOG_TAG ,__VA_ARGS__) // 定义LOGF类型
#endif

typedef struct {
    std::string cls_layer;
    std::string dis_layer;
    int stride;
} HeadInfo;

//typedef struct {
//    float x1;
//    float y1;
//    float x2;
//    float y2;
//    float score;
//    int label;
//} BoxInfo;

class NanoDet {
public:
    NanoDet(std::string proto, std::string model, bool useGPU);

    ~NanoDet();

    std::vector<BoxInfo> detect(JNIEnv *env, jobject bitmap, float threshold, float nms_threshold);

private:
    void decode_infer(const std::shared_ptr<TNN_NS::Mat>& cls_pred,
                      const std::shared_ptr<TNN_NS::Mat>& dis_pred,
                      int stride, float threshold, std::vector<std::vector<BoxInfo>> &results);

    BoxInfo disPred2Bbox(const float *&dfl_det, int label, float score, int x, int y, int stride);

    void nms(std::vector<BoxInfo> &input_boxes, float NMS_THRESH);

private:
    std::shared_ptr<tnn::TNN> net;
    std::shared_ptr<TNN_NS::Instance> instance;
    TNN_NS::DeviceType device_type;

    int in_n = 1;
    int in_c = 3;
    int in_w = 320;
    int in_h = 320;

    const int num_class = 80;
    const int reg_max = 7;

    std::vector<HeadInfo> heads_info{
            // cls_pred | dis_pred | stride
            {"792", "795", 8},
            {"814", "817", 16},
            {"836", "839", 32},
    };

    std::vector<std::string> labels{"person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
                                    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
                                    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
                                    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
                                    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
                                    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
                                    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
                                    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
                                    "hair drier", "toothbrush"};

public:
    static NanoDet *detector;
    static bool hasGPU;
    static bool toUseGPU;
};

template<typename _Tp>
int activation_function_softmax(const _Tp *src, _Tp *dst, int length);

inline float fast_exp(float x);

inline float sigmoid(float x);

#endif //NANODET_H
