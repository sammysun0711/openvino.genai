#include <stdio.h>

#include <filesystem>
#include <random>
#include <string>
#include <queue>

#include "stb_image.h"
#include "blip.hpp"
#include "gui.hpp"
#include "openvino/openvino.hpp"
#include "tinyfiledialogs.h"
#include "worker.hpp"


struct Column {
    float currentHeight;
    int index;

    bool operator<(const Column& other) const {
        return currentHeight > other.currentHeight;
    }
};

void ShowImagesWithBalancedHeight(const std::vector<ImageData>& images, float windowWidth, float xscale) {
    float maxImageWidth = 175 * xscale;  
    float padding = 10.0f * xscale; 

    int columnCount = static_cast<int>(windowWidth / (maxImageWidth + padding));
    if (columnCount < 1)
        columnCount = 1;

    std::priority_queue<Column> columns;
    for (int i = 0; i < columnCount; ++i) {
        columns.push({0.0f, i});
    }

    std::vector<std::vector<ImVec2>> columnLayouts(columnCount);
    std::vector<std::vector<ImageData>> columnImages(columnCount);

    for (const auto& image : images) {
        // calculate the new height
        float aspect_ratio = static_cast<float>(image.height) / image.width;
        float new_height = maxImageWidth * aspect_ratio;

        // lowest one
        Column shortestColumn = columns.top();
        columns.pop();

        columnLayouts[shortestColumn.index].emplace_back(maxImageWidth, new_height);
        columnImages[shortestColumn.index].push_back(image);

        shortestColumn.currentHeight += new_height + padding;
        columns.push(shortestColumn);
    }

    for (int col = 0; col < columnCount; ++col) {
        if (col > 0) {
            ImGui::SameLine();
        }
        ImGui::BeginGroup();  // keep alignment
        for (size_t i = 0; i < columnImages[col].size(); ++i) {
            const auto& image = columnImages[col][i];
            const auto& size = columnLayouts[col][i];

            ImGui::Image((void*)(intptr_t)image.textureID, size);

            if (ImGui::IsItemHovered()) {
                ImGui::BeginTooltip();
                ImGui::Text("[#%d]: %s",image.rank, image.path.c_str());
                ImGui::EndTooltip();
            }
        }
        ImGui::EndGroup();
    }
}

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
    const std::vector<std::string> required_subdirs = {"bge-small-zh-v1.5",
                                                       "blip_vqa_base",
                                                       "TinyLlama-1.1B-Chat-v1.0"};

    for (const std::string& subdir : required_subdirs) {
        std::filesystem::path subdir_path = std::filesystem::path(path) / subdir;
        if (!std::filesystem::exists(subdir_path) || !std::filesystem::is_directory(subdir_path)) {
            return false;
        }

        bool has_xml = false;
        bool has_bin = false;

        for (const auto& entry : std::filesystem::directory_iterator(subdir_path)) {
            if (entry.path().extension() == ".xml") {
                has_xml = true;
            } else if (entry.path().extension() == ".bin") {
                has_bin = true;
            }
            if (has_xml && has_bin) {
                break;
            }
        }

        if (!has_xml || !has_bin) {
            return false;
        }
    }

    return true;
}

std::string openFolderDialog() {
    const char* path = tinyfd_selectFolderDialog("Select Model Directory", "");
    if (path && validate_directory(path)) {
        return path;
    }
    return "";
}

void imgui_fix_text(ImFont* font, const char* text) {
    ImGui::PushFont(font);
    ImVec2 sz = ImGui::CalcTextSize(text);
    ImGui::PopFont();
    float canvasWidth = ImGui::GetWindowContentRegionMax().x - ImGui::GetWindowContentRegionMin().x;
    float origScale = font->Scale;
    font->Scale = canvasWidth / sz.x;
    ImGui::PushFont(font);
    ImGui::Text("%s", text);
    ImGui::PopFont();
    font->Scale = origScale;
}

int App::Init() {
    if (!glfwInit())
        return 1;

    glfwSetErrorCallback(glfw_error_callback);

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);

    window = glfwCreateWindow(1200, 800, "Rag Sample Image Search Demo", NULL, NULL);
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

    ImFontConfig config;
    font = io.Fonts->AddFontDefault(&config);

    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init(glsl_version);

    // init worker
    worker.Start();

    // init ov
    worker.Request([this] {
        ov::Core core;
        state.devices = core.get_available_devices();
        state.active_device_index = 0;
    });

    return 0;
}

int App::Clean() {
    worker.Stop();

    if (state.input_image.textureID)
        glDeleteTextures(1, &state.input_image.textureID);

    for (auto image : state.result_images) {
        if (image.textureID)
            glDeleteTextures(1, &image.textureID);
    }
    state.result_images.clear();


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

void App::LoadInputImageData() {
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

    if (state.input_image.textureID)
        glDeleteTextures(1, &state.input_image.textureID);

    state.input_image.data = stbi_load(state.input_image.path.c_str(),
                               &state.input_image.width,
                               &state.input_image.height,
                               &state.input_image.channels,
                               3);
    if (!state.input_image.data)
        throw std::runtime_error("failed to load image");

    glGenTextures(1, &state.input_image.textureID);
    glBindTexture(GL_TEXTURE_2D, state.input_image.textureID);
    glTexImage2D(GL_TEXTURE_2D,
                 0,
                 GL_RGB,
                 state.input_image.width,
                 state.input_image.height,
                 0,
                 GL_RGB,
                 GL_UNSIGNED_BYTE,
                 state.input_image.data);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    stbi_image_free(state.input_image.data);
}

void App::RenderLeftPanel() {
    ImGui::BeginChild("LeftPanel", ImVec2(600 * xscale, 0), true);
    ImGui::Text("Options");

    std::vector<const char*> items;
    for (int i = 0; i < state.devices.size(); ++i)
        items.push_back(state.devices[i].c_str());

    ImGui::Combo("Device", &state.active_device_index, items.data(), items.size());

    if (state.model_path.empty()) {
        ImGui::SameLine();
        if (ImGui::Button("Model")) {
            std::string model_path = openFolderDialog();
            if (!model_path.empty()) {
                state.model_path = model_path;
                worker.Request([this] {
                    // init server context
                    util::Args args;
                    auto device = state.devices[state.active_device_index];
                    args.image_embedding_device = device;
                    auto image_embedding_model_path = std::filesystem::path(state.model_path) / "blip_vqa_base";
                    args.image_embedding_model_path = image_embedding_model_path.u8string();
                    server_context = std::make_shared<util::ServerContext>(args);
                    worker.Request([this] {
                        handle_db_init(*server_context);
                        handle_image_embeddings_init((*server_context));
                    });
                });
            }
        }
    } else {
        ImGui::Text("Using model %s", state.model_path.c_str());
    }

    ImGui::SeparatorText("Image Search");
    if (ImGui::Button("Select Image")) {
        std::string next_image_path = openFileDialog();
        if (!next_image_path.empty()) {
            if (state.input_image.path != next_image_path) {
                state.input_image.path = next_image_path;
                state.should_load = true;
            }
        }
    }

    if (!state.input_image.path.empty()) {
        if (server_context != nullptr && is_image_embeddings_ready(*server_context)) {
            ImGui::SameLine();
            if (ImGui::InputInt("top k", &state.topk)){
                if (state.topk < 1) {
                    state.topk = 1;
                }
                if (state.topk > 30) {
                    state.topk = 30;
                }
            }
            ImGui::SameLine();
            ImGui::PushStyleColor(ImGuiCol_Button, (ImVec4)ImColor::HSV(5 / 7.0f, 0.6f, 0.6f));
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, (ImVec4)ImColor::HSV(5 / 7.0f, 0.7f, 0.7f));
            ImGui::PushStyleColor(ImGuiCol_ButtonActive, (ImVec4)ImColor::HSV(5 / 7.0f, 0.8f, 0.8f));

            if (ImGui::Button("Search")) {
                worker.Request([this] {
                    state.results.clear();
                    std::vector<std::string> inputs = {state.input_image.path};
                    state.results = handle_db_retrieval_image(*server_context, inputs, state.topk);
                    state.should_load_results = true;
                    for (size_t i = 0; i < state.results.size(); i++) {
                        std::cout << state.results[i] << std::endl;
                    }
                });
            }
            ImGui::PopStyleColor(3);
        }

        imgui_fix_text(font, state.input_image.path.c_str());
    }

    if (state.should_load) {
        // load image data and construct texture
        LoadInputImageData();
        state.should_load = false;
    }

    if (state.input_image.textureID) {
        float aspect_ratio = (float)state.input_image.width / state.input_image.height;
        ImVec2 preview_size(600 * xscale, 600 * yscale);

        if (aspect_ratio > 1.0f) {
            // Image is wider than tall
            preview_size.y = preview_size.x / aspect_ratio;
        } else {
            preview_size.x = preview_size.y * aspect_ratio;
        }

        ImVec2 padding = {(600.0f * xscale - preview_size.x) * 0.5f, (600.0f * yscale - preview_size.y) * 0.5f};
        ImVec2 window_pos = ImGui::GetCursorScreenPos();

        // Draw a black background
        ImGui::GetWindowDrawList()->AddRectFilled(window_pos,
                                                  ImVec2(window_pos.x + 600.0f * xscale, window_pos.y + 600.0f * yscale),
                                                  IM_COL32(128, 128, 128, 128));
        ImGui::SetCursorScreenPos(ImVec2(window_pos.x + padding.x, window_pos.y + padding.y));
        // Draw the texture
        ImGui::Image((void*)(intptr_t)state.input_image.textureID, preview_size);
        // Reset cursor to the next item after the 600x600 box
        ImGui::SetCursorScreenPos(ImVec2(window_pos.x, window_pos.y + 600.0f * yscale));
    }

    ImGui::EndChild();
}


void App::LoadResultImageData() {
    for (auto image : state.result_images) {
        if (image.textureID)
            glDeleteTextures(1, &image.textureID);
    }
    state.result_images.clear();

    for (size_t i = 0; i < state.results.size(); i++) {
        const auto path = state.results[i];
        ImageData imageData;
        imageData.path = path;
        imageData.rank = i + 1;

        imageData.data = stbi_load(path.c_str(), &imageData.width, &imageData.height, &imageData.channels, 3);
        if (!imageData.data)
            throw std::runtime_error("failed to load image");

        glGenTextures(1, &imageData.textureID);
        glBindTexture(GL_TEXTURE_2D, imageData.textureID);
        glTexImage2D(GL_TEXTURE_2D,
                     0,
                     GL_RGBA,
                     imageData.width,
                     imageData.height,
                     0,
                     imageData.channels == 4 ? GL_RGBA : GL_RGB,
                     GL_UNSIGNED_BYTE,
                     imageData.data);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        stbi_image_free(imageData.data);
        state.result_images.push_back(imageData);
    }
}

void App::RenderRightPanel() {
    ImGui::BeginChild("RightPanel", ImVec2(600 * xscale, 0), true);
    if (server_context == nullptr) {
        ImGui::Text("Loading Model..");
    } else {
        ImGui::Text("Ready");
    }

    if (state.should_load_results) {
        LoadResultImageData();
        state.should_load_results = false;
    }
    float windowWidth = ImGui::GetWindowSize().x;
    ShowImagesWithBalancedHeight(state.result_images, windowWidth, xscale);

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

    ImGui::Begin("Image Search Sample", nullptr, window_flags);

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