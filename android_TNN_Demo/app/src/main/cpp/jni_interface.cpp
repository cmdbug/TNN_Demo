#include <jni.h>
#include <string>
#include <fstream>

#include "tnn/core/tnn.h"
#include "tnn/core/blob.h"

#include "Yolov5.h"
#include "NanoDet.h"

#include <android/bitmap.h>
#include <android/log.h>
#include <opencv2/imgproc/types_c.h>

#ifndef LOG_TAG
#define LOG_TAG "WZT_TNN"
#define TLOGD(...) __android_log_print(ANDROID_LOG_DEBUG,LOG_TAG ,__VA_ARGS__) // 定义LOGD类型
#define TLOGI(...) __android_log_print(ANDROID_LOG_INFO,LOG_TAG ,__VA_ARGS__) // 定义LOGI类型
#define TLOGW(...) __android_log_print(ANDROID_LOG_WARN,LOG_TAG ,__VA_ARGS__) // 定义LOGW类型
#define TLOGE(...) __android_log_print(ANDROID_LOG_ERROR,LOG_TAG ,__VA_ARGS__) // 定义LOGE类型
#define TLOGF(...) __android_log_print(ANDROID_LOG_FATAL,LOG_TAG ,__VA_ARGS__) // 定义LOGF类型
#endif


JNIEXPORT jint JNI_OnLoad(JavaVM *vm, void *reserved) {
    return JNI_VERSION_1_6;
}

JNIEXPORT void JNI_OnUnload(JavaVM *vm, void *reserved) {
	delete YoloV5::detector;
	delete NanoDet::detector;
}

/* ======================================[ YOLOv5 ]======================================*/
extern "C"
JNIEXPORT void JNICALL
Java_com_wzt_tnn_model_YOLOv5_init(JNIEnv *env, jclass clazz, jstring proto, jstring model, jstring path, jboolean use_gpu) {
	if (YoloV5::detector != nullptr) {
        delete YoloV5::detector;
        YoloV5::detector = nullptr;
    }
    if (YoloV5::detector == nullptr) {
        std::string parentPath = env->GetStringUTFChars(path, 0);
        std::string protoPathStr = parentPath + env->GetStringUTFChars(proto, 0);
        std::string modelPathStr = parentPath + env->GetStringUTFChars(model, 0);
        YoloV5::detector = new YoloV5(protoPathStr, modelPathStr, use_gpu);
    }
}

extern "C"
JNIEXPORT jobjectArray JNICALL
Java_com_wzt_tnn_model_YOLOv5_detect(JNIEnv *env, jclass clazz, jobject bitmap, jbyteArray image_bytes, jint width,
                                     jint height, jdouble threshold, jdouble nms_threshold) {
    auto result = YoloV5::detector->detect(env, bitmap, threshold, nms_threshold);
    auto box_cls = env->FindClass("com/wzt/tnn/model/BoxInfo");
    auto cid = env->GetMethodID(box_cls, "<init>", "(FFFFIF)V");
    jobjectArray ret = env->NewObjectArray(result.size(), box_cls, nullptr);
    int i = 0;
    for (auto &box:result) {
        env->PushLocalFrame(1);
        jobject obj = env->NewObject(box_cls, cid, box.x1, box.y1, box.x2, box.y2, box.label, box.score);
        obj = env->PopLocalFrame(obj);
        env->SetObjectArrayElement(ret, i++, obj);
    }
    return ret;
}

/* ======================================[ NanoDet ]======================================*/
extern "C"
JNIEXPORT void JNICALL
Java_com_wzt_tnn_model_NanoDet_init(JNIEnv *env, jclass clazz, jstring proto, jstring model, jstring path, jboolean use_gpu) {
    if (NanoDet::detector != nullptr) {
        delete NanoDet::detector;
        NanoDet::detector = nullptr;
    }
    if (NanoDet::detector == nullptr) {
        std::string parentPath = env->GetStringUTFChars(path, 0);
        std::string protoPathStr = parentPath + env->GetStringUTFChars(proto, 0);
        std::string modelPathStr = parentPath + env->GetStringUTFChars(model, 0);
        NanoDet::detector = new NanoDet(protoPathStr, modelPathStr, use_gpu);
    }
}

extern "C"
JNIEXPORT jobjectArray JNICALL
Java_com_wzt_tnn_model_NanoDet_detect(JNIEnv *env, jclass clazz, jobject bitmap, jbyteArray image_bytes, jint width,
                                     jint height, jdouble threshold, jdouble nms_threshold) {
    auto result = NanoDet::detector->detect(env, bitmap, threshold, nms_threshold);
    auto box_cls = env->FindClass("com/wzt/tnn/model/BoxInfo");
    auto cid = env->GetMethodID(box_cls, "<init>", "(FFFFIF)V");
    jobjectArray ret = env->NewObjectArray(result.size(), box_cls, nullptr);
    int i = 0;
    for (auto &box:result) {
        env->PushLocalFrame(1);
        jobject obj = env->NewObject(box_cls, cid, box.x1, box.y1, box.x2, box.y2, box.label, box.score);
        obj = env->PopLocalFrame(obj);
        env->SetObjectArrayElement(ret, i++, obj);
    }
    return ret;
}


