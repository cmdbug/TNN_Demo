package com.wzt.tnn.model;

import android.graphics.Bitmap;

public class NanoDet {

    static {
        System.loadLibrary("tengtnn");
    }

    public static native void init(String proto, String model, String path, boolean useGPU);
    public static native BoxInfo[] detect(Bitmap bitmap, byte[] imageBytes, int width, int height, double threshold, double nms_threshold);
}
