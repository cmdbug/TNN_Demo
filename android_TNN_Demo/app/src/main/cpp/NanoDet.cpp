#include "NanoDet.h"
#include "omp.h"
#include <fstream>
#include <android/bitmap.h>

#include <opencv2/core/mat.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgproc.hpp>

NanoDet *NanoDet::detector = nullptr;
bool NanoDet::hasGPU = true;
bool NanoDet::toUseGPU = true;

NanoDet::NanoDet(std::string proto, std::string model, bool useGPU) {
    if (NanoDet::net == nullptr) {
        std::string protoContent, modelContent;
        protoContent = fdLoadFile(proto);  // see yolov5
        modelContent = fdLoadFile(model);  // see yolov5

        TNN_NS::Status status;
        TNN_NS::ModelConfig config;
        config.model_type = TNN_NS::MODEL_TYPE_TNN;
        config.params = {protoContent, modelContent};
        auto net = std::make_shared<TNN_NS::TNN>();
        status = net->Init(config);
        NanoDet::net = net;

        NanoDet::device_type = useGPU ? TNN_NS::DEVICE_OPENCL : TNN_NS::DEVICE_ARM;

        TNN_NS::InputShapesMap shapeMap;
        TNN_NS::NetworkConfig network_config;
        network_config.library_path = {""};
        network_config.device_type = NanoDet::device_type;
        auto ins = NanoDet::net->CreateInst(network_config, status, shapeMap);
        if (status != TNN_NS::TNN_OK || !ins) {
            TLOGW("GPU initialization failed, switch to CPU");
            // 如果出现GPU加载失败，切换到CPU
            NanoDet::device_type = TNN_NS::DEVICE_ARM;
            network_config.device_type = TNN_NS::DEVICE_ARM;
            ins = NanoDet::net->CreateInst(network_config, status, shapeMap);
        }
        NanoDet::instance = ins;

        if (status != TNN_NS::TNN_OK) {
            TLOGE("TNN init failed %d", (int) status);
            return;
        }
    }
}

NanoDet::~NanoDet() {
    NanoDet::instance = nullptr;
    NanoDet::net = nullptr;
}

std::vector<BoxInfo> NanoDet::detect(JNIEnv *env, jobject bitmap, float threshold, float nms_threshold) {
    std::vector<BoxInfo> results;
    AndroidBitmapInfo bitmapInfo;
    void *imageSource;
    if (AndroidBitmap_getInfo(env, bitmap, &bitmapInfo) < 0) {
        return results;
    }
    if (bitmapInfo.format != ANDROID_BITMAP_FORMAT_RGBA_8888) {
        return results;
    }
    if (AndroidBitmap_lockPixels(env, bitmap, &imageSource) < 0) {
        return results;
    }
    int image_h = bitmapInfo.height;
    int image_w = bitmapInfo.width;
    float scale_w = 1.0f * in_w / image_w;
    float scale_h = 1.0f * in_h / image_h;

    // 格式转换 RGBA -> BGRA
    cv::Mat dst(image_h, image_w, CV_8UC3);
    cv::Mat src(image_h, image_w, CV_8UC4, imageSource);
    cvtColor(src, dst, cv::COLOR_RGBA2BGRA);

    // 原始图片
    TNN_NS::DeviceType dt = TNN_NS::DEVICE_ARM;  // 当前数据来源始终位于CPU，不需要设置成OPENCL，tnn自动复制cpu->gpu
    TNN_NS::DimsVector image_dims = {1, 4, image_h, image_w};
    auto input_mat = std::make_shared<TNN_NS::Mat>(dt, TNN_NS::N8UC4, image_dims, dst.data);  // imageSource(RGBA) or dst.data(BGR)
    // 模型输入
    TNN_NS::DimsVector target_dims = {1, 4, in_h, in_w};
    auto resize_mat = std::make_shared<TNN_NS::Mat>(dt, TNN_NS::N8UC4, target_dims);
    // opencl需要设置queue
    void *command_queue = nullptr;
    auto status = NanoDet::instance->GetCommandQueue(&command_queue);
    if (status != TNN_NS::TNN_OK) {
        TLOGE("MatUtils::GetCommandQueue Error: %s", status.description().c_str());
    }
    // 转换大小
    TNN_NS::ResizeParam param;
    param.type = TNN_NS::InterpType::INTERP_TYPE_NEAREST;
    status = TNN_NS::MatUtils::Resize(*input_mat, *resize_mat, param, command_queue);
    if (status != TNN_NS::TNN_OK) {
        TLOGE("MatUtils::Resize Error: %s", status.description().c_str());
    }

    // 输入数据
    TNN_NS::MatConvertParam input_cvt_param;
    input_cvt_param.scale = {0.017429f, 0.017507f, 0.017125f, 0.0};
    input_cvt_param.bias = {-(103.53f * 0.017429f), -(116.28f * 0.017507f), -(123.675f * 0.017125f), 0.0};
    status = NanoDet::instance->SetInputMat(resize_mat, input_cvt_param);
    if (status != TNN_NS::TNN_OK) {
        TLOGE("instance.SetInputMat Error: %s", status.description().c_str());
    }

    // 前向
//    TNN_NS::Callback callback;
//    status = NanoDet::instance->ForwardAsync(callback);
    status = NanoDet::instance->Forward();
    if (status != TNN_NS::TNN_OK) {
        TLOGE("instance.Forward Error: %s", status.description().c_str());
    }

    std::vector<std::vector<BoxInfo>> resultx;
    resultx.resize(num_class);
    // 获取数据
    for (const auto &head_info : heads_info) {
        TNN_NS::MatConvertParam clsPparam;
        std::shared_ptr<TNN_NS::Mat> cls_pred;
        status = NanoDet::instance->GetOutputMat(cls_pred, clsPparam, head_info.cls_layer);
        if (status != TNN_NS::TNN_OK) {
            TLOGE("instance.GetOutputMat Error:cls_pred %s", status.description().c_str());
        }
        //TLOGD("===============> %d %d %d %d", cls_pred->GetDims()[0], cls_pred->GetDims()[1], cls_pred->GetDims()[2], cls_pred->GetDims()[3]);

        TNN_NS::MatConvertParam disParam;
        std::shared_ptr<TNN_NS::Mat> dis_pred;
        status = NanoDet::instance->GetOutputMat(dis_pred, disParam, head_info.dis_layer);
        if (status != TNN_NS::TNN_OK) {
            TLOGE("instance.GetOutputMat Error:dis_pred %s", status.description().c_str());
        }
        //TLOGD("===============> %d %d %d %d", dis_pred->GetDims()[0], dis_pred->GetDims()[1], dis_pred->GetDims()[2], dis_pred->GetDims()[3]);

        // 后处理
        decode_infer(cls_pred, dis_pred, head_info.stride, threshold, resultx);
    }

    for (int i = 0; i < resultx.size(); i++) {
        nms(resultx[i], nms_threshold);
        for (auto box : resultx[i]) {
            box.x1 = box.x1 / scale_w;
            box.x2 = box.x2 / scale_w;
            box.y1 = box.y1 / scale_h;
            box.y2 = box.y2 / scale_h;
            results.push_back(box);
        }
    }

    TLOGD("object size:%ld", results.size());

    AndroidBitmap_unlockPixels(env, bitmap);
    if (status != TNN_NS::TNN_OK) {
        TLOGE("get outputmat fail");
        return results;
    }

    return results;
}

void NanoDet::decode_infer(const std::shared_ptr<TNN_NS::Mat> &cls_pred, const std::shared_ptr<TNN_NS::Mat> &dis_pred,
                           int stride, float threshold, std::vector<std::vector<BoxInfo>> &results) {
    int feature_h = in_h / stride;
    int feature_w = in_w / stride;

    TNN_NS::MatType matType = cls_pred->GetMatType();  // NCHW_float
    std::vector<int> dims = cls_pred->GetDims();
    //cv::Mat debug_heatmap = cv::Mat(feature_h, feature_w, CV_8UC3);
    for (int idx = 0; idx < feature_h * feature_w; idx++) {
        // scores is a tensor with shape [feature_h * feature_w, num_class]
//        const float *scores = cls_pred->host<float>() + (idx * num_class);
        auto *scoreTemp = static_cast<float *>(cls_pred->GetData());
        const float *scores = scoreTemp + (idx * num_class);

        int row = idx / feature_w;
        int col = idx % feature_w;
        float score = 0;
        int cur_label = 0;
        for (int label = 0; label < num_class; label++) {
            if (scores[label] > score) {
                score = scores[label];
                cur_label = label;
            }
        }
        if (score > threshold && score <= 1.0) {
            TLOGD("score: %f  %s", score, labels[cur_label].c_str());
            // bbox is a tensor with shape [feature_h * feature_w, 4_points * 8_distribution_bite]
//            const float *bbox_pred = dis_pred->host<float>() + (idx * 4 * (reg_max + 1));
            auto *bboxTemp = static_cast<float *>(dis_pred->GetData());
            const float *bbox_pred = bboxTemp + (idx * 4 * (reg_max + 1));
            BoxInfo boxInfo = disPred2Bbox(bbox_pred, cur_label, score, col, row, stride);
            results[cur_label].push_back(boxInfo);
            //debug_heatmap.at<cv::Vec3b>(row, col)[0] = 255;
        }
    }
}

BoxInfo NanoDet::disPred2Bbox(const float *&dfl_det, int label, float score, int x, int y, int stride) {
    float ct_x = (x + 0.5f) * stride;
    float ct_y = (y + 0.5f) * stride;
    std::vector<float> dis_pred;
    dis_pred.resize(4);
    for (int i = 0; i < 4; i++) {
        float dis = 0;
        auto *dis_after_sm = new float[reg_max + 1];
        activation_function_softmax(dfl_det + i * (reg_max + 1), dis_after_sm, reg_max + 1);
        for (int j = 0; j < reg_max + 1; j++) {
            dis += j * dis_after_sm[j];
        }
        dis *= stride;
        dis_pred[i] = dis;
        delete[] dis_after_sm;
    }
    float xmin = (std::max)(ct_x - dis_pred[0], .0f);
    float ymin = (std::max)(ct_y - dis_pred[1], .0f);
    float xmax = (std::min)(ct_x + dis_pred[2], (float) in_w);
    float ymax = (std::min)(ct_y + dis_pred[3], (float) in_h);

    return BoxInfo{xmin, ymin, xmax, ymax, score, label};
}

void NanoDet::nms(std::vector<BoxInfo> &input_boxes, float NMS_THRESH) {
    std::sort(input_boxes.begin(), input_boxes.end(),
              [](BoxInfo a, BoxInfo b) { return a.score > b.score; }
    );
    std::vector<float> vArea(input_boxes.size());
    for (int i = 0; i < int(input_boxes.size()); ++i) {
        vArea[i] = (input_boxes.at(i).x2 - input_boxes.at(i).x1 + 1)
                   * (input_boxes.at(i).y2 - input_boxes.at(i).y1 + 1);
    }
    for (int i = 0; i < int(input_boxes.size()); ++i) {
        for (int j = i + 1; j < int(input_boxes.size());) {
            float xx1 = (std::max)(input_boxes[i].x1, input_boxes[j].x1);
            float yy1 = (std::max)(input_boxes[i].y1, input_boxes[j].y1);
            float xx2 = (std::min)(input_boxes[i].x2, input_boxes[j].x2);
            float yy2 = (std::min)(input_boxes[i].y2, input_boxes[j].y2);
            float w = (std::max)(float(0), xx2 - xx1 + 1);
            float h = (std::max)(float(0), yy2 - yy1 + 1);
            float inter = w * h;
            float ovr = inter / (vArea[i] + vArea[j] - inter);
            if (ovr >= NMS_THRESH) {
                input_boxes.erase(input_boxes.begin() + j);
                vArea.erase(vArea.begin() + j);
            } else {
                j++;
            }
        }
    }
}

inline float fast_exp(float x) {
    union {
        uint32_t i;
        float f;
    } v{};
    v.i = (1 << 23) * (1.4426950409 * x + 126.93490512f);
    return v.f;
}

inline float sigmoid(float x) {
    return 1.0f / (1.0f + fast_exp(-x));
}

template<typename _Tp>
int activation_function_softmax(const _Tp *src, _Tp *dst, int length) {
    const _Tp alpha = *std::max_element(src, src + length);
    _Tp denominator{0};

    for (int i = 0; i < length; ++i) {
        dst[i] = fast_exp(src[i] - alpha);
        denominator += dst[i];
    }

    for (int i = 0; i < length; ++i) {
        dst[i] /= denominator;
    }

    return 0;
}
