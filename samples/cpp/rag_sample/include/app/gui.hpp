// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include <GLFW/glfw3.h>

#include <string>
#include <cstdint>

#include "worker.hpp"

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
    //std::shared_ptr<StableDiffusionControlnetPipeline> pipe = nullptr;
    Worker worker;

    std::atomic<bool> running{false};
    float xscale = 1.0;
    float yscale = 1.0;
};