// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include <GLFW/glfw3.h>
#include "blip.hpp"
#include "server_context.hpp"

#include <string>
#include <cstdint>

#include "worker.hpp"

struct ImageData {
    unsigned char* data;
    int width, height, channels;
    GLuint textureID;
    std::string path;
    int rank;
};

struct UIState {
    ImageData input_image;
    bool should_load = false;

    std::vector<std::string> devices;
    int active_device_index = 0;
    std::string model_path;
    int topk = 10;
    bool debug = false;
    std::vector<std::string> results;
    bool should_load_results = false;
    std::vector<ImageData> result_images;
};

class App {
public:
    int Init();
    int Run();

private:
    void RenderLeftPanel();
    void RenderRightPanel();
    void Render();
    int Clean();
    void LoadInputImageData();
    void LoadResultImageData();

    const char* glsl_version = "#version 130";
    GLFWwindow* window;
    UIState state;
    std::shared_ptr<util::ServerContext> server_context = nullptr;
    Worker worker;

    std::atomic<bool> running{false};
    float xscale = 1.0;
    float yscale = 1.0;

    ImFont* font;
};