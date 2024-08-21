#include <stdio.h>

#include <filesystem>
#include <random>
#include <string>

#include "gui.hpp"
#include "tinyfiledialogs.h"
#include "worker.hpp"

static void glfw_error_callback(int error, const char* description) {
    fprintf(stderr, "Glfw Error %d: %s\n", error, description);
}

std::string openFileDialog() {
    const char* filters[] = {"*.png", "*.jpg", "*.jpeg", "*.bmp"};
    const char* filePath = tinyfd_openFileDialog("Select an Image",  // Dialog title
                                                 "",                 // Default path
                                                 1,                  // Number of filters
                                                 filters,            // Filters
                                                 "Image files",      // Filter description
                                                 1                   // Single selection
    );

    return filePath ? filePath : "";  // Return empty string if canceled
}

bool validate_directory(const std::string& path) {
    bool has_xml = false;
    bool has_bin = false;

    for (const auto& entry : std::filesystem::directory_iterator(path)) {
        if (entry.path().extension() == ".xml") {
            has_xml = true;
        } else if (entry.path().extension() == ".bin") {
            has_bin = true;
        }
        if (has_xml && has_bin) {
            return true;
        }
    }
    return false;
}

std::string openFolderDialog() {
    const char* path = tinyfd_selectFolderDialog("Select Model Directory", "");
    if (path && validate_directory(path)) {
        return path;
    }
    return "";
}

int App::Init() {
    if (!glfwInit())
        return 1;

    glfwSetErrorCallback(glfw_error_callback);

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);

    window = glfwCreateWindow(1200, 800, "Stable Diffusion Controlnet Demo", NULL, NULL);
    if (window == NULL)
        return 1;
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);  // Enable vsync

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui::StyleColorsDark();

    glfwGetWindowContentScale(window, &xscale, &yscale);

    // resize window
    int width, height;
    glfwGetWindowSize(window, &width, &height);
    int new_width = static_cast<int>(width * xscale);
    int new_height = static_cast<int>(height * yscale);
    glfwSetWindowSize(window, new_width, new_height);

    // resize fonts and all
    ImGuiIO& io = ImGui::GetIO();
    io.Fonts->Clear();
    io.Fonts->AddFontDefault();
    io.FontGlobalScale = xscale;
    ImGui::GetStyle().ScaleAllSizes(xscale);

    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init(glsl_version);

    // init ov
    // ov::Core core;
    // state.devices = core.get_available_devices();
    // state.active_device_index = 0;

    // init worker
    worker.Start();

    return 0;
}

int App::Clean() {
    worker.Stop();

    // if (preview_state.preview_texture)
    //     glDeleteTextures(1, &preview_state.preview_texture);

    // if (result_state.texture)
    //     glDeleteTextures(1, &result_state.texture);

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}

int App::Run() {
    while (!glfwWindowShouldClose(window)) {
        Render();
    }

    Clean();
    return 0;
}

void App::RenderLeftPanel() {
    ImGui::BeginChild("LeftPanel", ImVec2(600 * xscale, 0), true);
    ImGui::Text("Options");

    
    ImGui::EndChild();
}

void App::RenderRightPanel() {
    ImGui::BeginChild("RightPanel", ImVec2(600 * xscale, 0), true);
    ImGui::Text("Results");

    ImGui::EndChild();
}



void App::Render() {
    glfwPollEvents();

    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    int display_w, display_h;
    glfwGetFramebufferSize(window, &display_w, &display_h);

    ImGui::SetNextWindowPos(ImVec2(0, 0));
    ImGui::SetNextWindowSize(ImVec2(display_w, display_h));

    ImGuiWindowFlags window_flags = 0;
    window_flags |= ImGuiWindowFlags_NoTitleBar;
    window_flags |= ImGuiWindowFlags_NoResize;
    window_flags |= ImGuiWindowFlags_NoMove;
    window_flags |= ImGuiWindowFlags_NoCollapse;
    window_flags |= ImGuiWindowFlags_NoBringToFrontOnFocus;

    ImGui::Begin("Stable Diffusion Controlnet", nullptr, window_flags);

    RenderLeftPanel();
    ImGui::SameLine();
    RenderRightPanel();

    ImGui::End();

    ImGui::Render();

    glfwGetFramebufferSize(window, &display_w, &display_h);
    glViewport(0, 0, display_w, display_h);
    glClear(GL_COLOR_BUFFER_BIT);
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

    glfwSwapBuffers(window);
}

int main(int, char**) {
    App app;

    if (app.Init()) {
        fprintf(stderr, "Failed to init app\n");
        return -1;
    }

    return app.Run();
}