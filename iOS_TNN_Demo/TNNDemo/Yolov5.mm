//
//  ViewController.h
//  TNNDemo
//
//  Created by WZTENG on 2021/1/11.
//  Copyright © 2021 TENG. All rights reserved.
//

#include "Yolov5.h"

YoloV5 *YoloV5::detector = nullptr;
bool YoloV5::hasGPU = true;
bool YoloV5::toUseGPU = true;

std::shared_ptr<ModelInfo> loadModelToInfo() {
    auto modelInfo = std::make_shared<ModelInfo>();
    
    auto library_bundle_path = [[NSBundle mainBundle] pathForResource:@"tnn" ofType:@"bundle"];
    auto library_bundle = [NSBundle bundleWithPath:library_bundle_path];
    auto library_path = [library_bundle pathForResource:@"tnn" ofType:@"metallib"];

    auto proto_path = [[NSBundle mainBundle] pathForResource:@"yolov5s_sim_opt.tnnproto" ofType:nil];
    auto model_path = [[NSBundle mainBundle] pathForResource:@"yolov5s_sim_opt.tnnmodel" ofType:nil];
    if (proto_path.length <= 0 || model_path.length <= 0) {
        NSLog(@"Error: proto or model path is invalid");
        return modelInfo;
    }
    std::string proto_content = [NSString stringWithContentsOfFile:proto_path encoding:NSUTF8StringEncoding error:nil].UTF8String;
    NSData *data_mode = [NSData dataWithContentsOfFile:model_path];
    std::string model_content = [data_mode length] > 0 ? std::string((const char *)[data_mode bytes], [data_mode length]) : "";
    if (proto_content.size() <= 0 || model_content.size() <= 0) {
        NSLog(@"Error: proto or model path is invalid");
        return modelInfo;
    }
    modelInfo->library_path = library_path.UTF8String;
    modelInfo->proto_content = proto_content;
    modelInfo->model_content = model_content;
    
    return modelInfo;
}

YoloV5::YoloV5(bool useGPU) {
    if (YoloV5::net == nullptr) {
        // 加载模型
        auto modelInfo = loadModelToInfo();

        TNN_NS::Status status;
        TNN_NS::ModelConfig config;
        config.model_type = TNN_NS::MODEL_TYPE_TNN;
        config.params = {modelInfo->proto_content, modelInfo->model_content};
        auto net = std::make_shared<TNN_NS::TNN>();
        status = net->Init(config);
        YoloV5::net = net;

        YoloV5::device_type = useGPU ? TNN_NS::DEVICE_METAL : TNN_NS::DEVICE_ARM;

        TNN_NS::InputShapesMap shapeMap;
        TNN_NS::NetworkConfig network_config;
        network_config.library_path = {modelInfo->library_path};
        network_config.device_type = YoloV5::device_type;
        auto ins = YoloV5::net->CreateInst(network_config, status, shapeMap);
        if (status != TNN_NS::TNN_OK || !ins) {
            NSLog(@"GPU initialization failed, switch to CPU");
            // 如果出现GPU加载失败，切换到CPU
            YoloV5::device_type = TNN_NS::DEVICE_ARM;
            network_config.device_type = TNN_NS::DEVICE_ARM;
            ins = YoloV5::net->CreateInst(network_config, status, shapeMap);
        }
        YoloV5::instance = ins;

        if (status != TNN_NS::TNN_OK) {
            NSLog(@"TNN init failed %d", (int) status);
            return;
        }
    }
}

YoloV5::~YoloV5() {
    YoloV5::instance = nullptr;
    YoloV5::net = nullptr;
}

std::vector<BoxInfo> YoloV5::detect(UIImage *image, float threshold, float nms_threshold) {
    std::vector<BoxInfo> results;
    
    auto imageSource = utility::UIImageGetData(image);
    
    int image_h = image.size.height;
    int image_w = image.size.width;
    float scale_w = 1.0f * net_width / image_w;
    float scale_h = 1.0f * net_height / image_h;
    // 原始图片
    TNN_NS::DeviceType dt = TNN_NS::DEVICE_ARM;  // 当前数据来源始终位于CPU，不需要设置成METAL，tnn自动复制cpu->gpu
    TNN_NS::DimsVector image_dims = {1, 4, image_h, image_w};
    auto input_mat = std::make_shared<TNN_NS::Mat>(dt, TNN_NS::N8UC4, image_dims, imageSource.get());
    // 模型输入
    TNN_NS::DimsVector target_dims = {1, 4, net_height, net_width};
    auto resize_mat = std::make_shared<TNN_NS::Mat>(dt, TNN_NS::N8UC4, target_dims);
	// OPENCL需要设置queue
    void *command_queue = nullptr;
    auto status = YoloV5::instance->GetCommandQueue(&command_queue);
    if (status != TNN_NS::TNN_OK) {
        NSLog(@"MatUtils::GetCommandQueue Error: %s", status.description().c_str());
    }
    // 转换大小
    TNN_NS::ResizeParam param;
    param.type = TNN_NS::InterpType::INTERP_TYPE_NEAREST;
    status = TNN_NS::MatUtils::Resize(*input_mat, *resize_mat, param, command_queue);
    if (status != TNN_NS::TNN_OK) {
        NSLog(@"MatUtils::Resize Error: %s", status.description().c_str());
    }

    // 输入数据
    TNN_NS::MatConvertParam input_cvt_param;
    input_cvt_param.scale = {1.0 / 255, 1.0 / 255, 1.0 / 255, 0.0};
    input_cvt_param.bias = {0.0, 0.0, 0.0, 0.0};
    status = YoloV5::instance->SetInputMat(resize_mat, input_cvt_param);
    if (status != TNN_NS::TNN_OK) {
        NSLog(@"instance.SetInputMat Error: %s", status.description().c_str());
    }

    // 前向
//    TNN_NS::Callback callback;
//    status = YoloV5::instance->ForwardAsync(callback);
    status = YoloV5::instance->Forward();
    if (status != TNN_NS::TNN_OK) {
        NSLog(@"instance.Forward Error: %s", status.description().c_str());
    }

    // 获取数据
    std::vector<std::shared_ptr<TNN_NS::Mat>> output_mats;
    for (const YoloLayerData &layerData : YoloV5::layers) {
        std::shared_ptr<TNN_NS::Mat> output_mat;
        TNN_NS::MatConvertParam param;
        status = YoloV5::instance->GetOutputMat(output_mat, param, layerData.name);
        if (status != TNN_NS::TNN_OK) {
            NSLog(@"instance.GetOutputMat Error: %s", status.description().c_str());
        }
        output_mats.push_back(output_mat);
    }

    // 后处理
    generateDetectResult(output_mats, results, threshold, nms_threshold);
    // 简单缩放回原图
    for (BoxInfo &boxInfo : results) {
        boxInfo.x1 = boxInfo.x1 / scale_w;
        boxInfo.x2 = boxInfo.x2 / scale_w;
        boxInfo.y1 = boxInfo.y1 / scale_h;
        boxInfo.y2 = boxInfo.y2 / scale_h;
    }
//    NSLog(@"object size:%ld", results.size());

    if (status != TNN_NS::TNN_OK) {
        NSLog(@"get outputmat fail");
        return results;
    }

    return results;
}

void YoloV5::generateDetectResult(std::vector<std::shared_ptr<TNN_NS::Mat>> outputs, std::vector<BoxInfo> &detecs,
                                  float threshold, float nms_threshold) {
    int blob_index = 0;
    int num_anchor = YoloV5::layers[0].anchors.size(); // 3
    int detect_dim = 0;  // 85
    for (auto &output:outputs) {
        auto dim = output->GetDims();

        detect_dim = dim[3] / num_anchor;
        if (dim[3] != num_anchor * detect_dim) {
            NSLog(@"Invalid detect output, the size of last dimension is: %d\n", dim[3]);
            return;
        }
        auto *data = static_cast<float *>(output->GetData());

        int num_potential_detecs = dim[1] * dim[2] * num_anchor;
        for (int i = 0; i < num_potential_detecs; ++i) {
            float x = data[i * detect_dim + 0];
            float y = data[i * detect_dim + 1];
            float width = data[i * detect_dim + 2];
            float height = data[i * detect_dim + 3];

            float objectness = data[i * detect_dim + 4];
            if (objectness < threshold) {
                continue;
            }

            // 计算坐标
            x = (float) (x * 2 - 0.5 + ((i / num_anchor) % dim[2])) * YoloV5::layers[blob_index].stride;
            y = (float) (y * 2 - 0.5 + ((i / num_anchor) / dim[2]) % dim[1]) * YoloV5::layers[blob_index].stride;
            width = pow((width * 2), 2) * YoloV5::layers[blob_index].anchors[i % num_anchor].width;
            height = pow((height * 2), 2) * YoloV5::layers[blob_index].anchors[i % num_anchor].height;
            // 坐标格式转换
            float x1 = x - width / 2;
            float y1 = y - height / 2;
            float x2 = x + width / 2;
            float y2 = y + height / 2;
            // 置信度
            auto conf_start = data + i * detect_dim + 5;
            auto conf_end = data + (i + 1) * detect_dim;
            auto max_conf_iter = std::max_element(conf_start, conf_end);
            int conf_idx = static_cast<int>(std::distance(conf_start, max_conf_iter));
            float score = (*max_conf_iter) * objectness;

            BoxInfo obj_info;
            obj_info.x1 = x1;
            obj_info.y1 = y1;
            obj_info.x2 = x2;
            obj_info.y2 = y2;
            obj_info.score = score;
            obj_info.label = conf_idx;
            detecs.push_back(obj_info);
        }
        blob_index += 1;
    }
    nms(detecs, nms_threshold);
}

void YoloV5::nms(std::vector<BoxInfo> &input_boxes, float nms_thresh) {
    std::sort(input_boxes.begin(), input_boxes.end(), [](BoxInfo a, BoxInfo b) { return a.score > b.score; });
    std::vector<float> vArea(input_boxes.size());
    for (int i = 0; i < int(input_boxes.size()); ++i) {
        vArea[i] = (input_boxes.at(i).x2 - input_boxes.at(i).x1 + 1)
                   * (input_boxes.at(i).y2 - input_boxes.at(i).y1 + 1);
    }
    for (int i = 0; i < int(input_boxes.size()); ++i) {
        for (int j = i + 1; j < int(input_boxes.size());) {
            float xx1 = std::max(input_boxes[i].x1, input_boxes[j].x1);
            float yy1 = std::max(input_boxes[i].y1, input_boxes[j].y1);
            float xx2 = std::min(input_boxes[i].x2, input_boxes[j].x2);
            float yy2 = std::min(input_boxes[i].y2, input_boxes[j].y2);
            float w = std::max(float(0), xx2 - xx1 + 1);
            float h = std::max(float(0), yy2 - yy1 + 1);
            float inter = w * h;
            float ovr = inter / (vArea[i] + vArea[j] - inter);
            if (ovr >= nms_thresh) {
                input_boxes.erase(input_boxes.begin() + j);
                vArea.erase(vArea.begin() + j);
            } else {
                j++;
            }
        }
    }
}
