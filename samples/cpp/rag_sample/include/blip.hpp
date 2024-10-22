// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <openvino/openvino.hpp>

class BlipModel {
public:
    ov::Core core;
    ov::InferRequest vision_model;
    ov::InferRequest vision_projection_model;

    BlipModel() = default;
    ~BlipModel() = default;

    void init(std::string blip_model_path, std::string device);

    std::vector<std::vector<float>> encode_images(std::vector<std::string> image_paths);

private:
    std::vector<float> encode_image(std::string image_path);
    ov::Tensor preprocess(std::string image_path);
};

// copy from https://github.com/wenyi5608/openvino.genai/blob/wenyi5608-stateful/llm/mincpmv2_cpp/clip.h
// RGB uint8 image
struct clip_image_u8 {
    int nx;
    int ny;

    std::vector<uint8_t> buf;
};

// RGB float32 image (NHWC)
// Memory layout: RGBRGBRGB...
struct clip_image_f32 {
    int nx;
    int ny;

    std::vector<float> buf;
};

bool clip_image_load_from_file(const char* fname, struct clip_image_u8* img);
bool bicubic_resize(const clip_image_u8& img, clip_image_u8& dst, int target_width, int target_height);
void normalize_image_u8_to_f32(const clip_image_u8* src, clip_image_f32* dst, const float mean[3], const float std[3]);