#include "blip.hpp"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include <filesystem>
#include <stdexcept>

const int BLIP_PREPROCESS_DEFAULT_W = 384;
const int BLIP_PREPROCESS_DEFAULT_H = 384;
const float OPENAI_CLIP_MEAN[3] = {0.48145466, 0.4578275, 0.40821073};
const float OPENAI_CLIP_STD[3] = {0.26862954, 0.26130258, 0.27577711};

inline float clamp(float x, float lower, float upper) {
    return std::max(lower, std::min(x, upper));

}
static void build_clip_img_from_data(const stbi_uc* data, int nx, int ny, clip_image_u8* img) {
    img->nx = nx;
    img->ny = ny;
    img->buf.resize(3 * nx * ny);
    memcpy(img->buf.data(), data, img->buf.size());
}

bool clip_image_load_from_file(const char* fname, clip_image_u8* img) {
    int nx, ny, nc;
    auto* data = stbi_load(fname, &nx, &ny, &nc, 3);
    if (!data) {
        std::cerr << "failed to load image: " << fname << '\n';
        return false;
    }
    build_clip_img_from_data(data, nx, ny, img);
    stbi_image_free(data);
    return true;
}

bool bicubic_resize(const clip_image_u8& img, clip_image_u8& dst, int target_width, int target_height) {
    const int nx = img.nx;
    const int ny = img.ny;

    dst.nx = target_width;
    dst.ny = target_height;
    dst.buf.resize(3 * target_width * target_height);

    float Cc;
    float C[5];
    float d0, d2, d3, a0, a1, a2, a3;
    int i, j, k, jj;
    int x, y;
    float dx, dy;
    float tx, ty;

    tx = (float)nx / (float)target_width;
    ty = (float)ny / (float)target_height;

    // Bicubic interpolation; adapted from ViT.cpp, inspired from :
    //    -> https://github.com/yglukhov/bicubic-interpolation-image-processing/blob/master/libimage.c#L36
    //    -> https://en.wikipedia.org/wiki/Bicubic_interpolation

    for (i = 0; i < target_height; i++) {
        for (j = 0; j < target_width; j++) {
            x = (int)(tx * j);
            y = (int)(ty * i);

            dx = tx * j - x;
            dy = ty * i - y;

            for (k = 0; k < 3; k++) {
                for (jj = 0; jj <= 3; jj++) {
                    d0 = img.buf[(clamp(y - 1 + jj, 0, ny - 1) * nx + clamp(x - 1, 0, nx - 1)) * 3 + k] -
                         img.buf[(clamp(y - 1 + jj, 0, ny - 1) * nx + clamp(x, 0, nx - 1)) * 3 + k];
                    d2 = img.buf[(clamp(y - 1 + jj, 0, ny - 1) * nx + clamp(x + 1, 0, nx - 1)) * 3 + k] -
                         img.buf[(clamp(y - 1 + jj, 0, ny - 1) * nx + clamp(x, 0, nx - 1)) * 3 + k];
                    d3 = img.buf[(clamp(y - 1 + jj, 0, ny - 1) * nx + clamp(x + 2, 0, nx - 1)) * 3 + k] -
                         img.buf[(clamp(y - 1 + jj, 0, ny - 1) * nx + clamp(x, 0, nx - 1)) * 3 + k];
                    a0 = img.buf[(clamp(y - 1 + jj, 0, ny - 1) * nx + clamp(x, 0, nx - 1)) * 3 + k];

                    a1 = -1.0 / 3 * d0 + d2 - 1.0 / 6 * d3;
                    a2 = 1.0 / 2 * d0 + 1.0 / 2 * d2;
                    a3 = -1.0 / 6 * d0 - 1.0 / 2 * d2 + 1.0 / 6 * d3;

                    C[jj] = a0 + a1 * dx + a2 * dx * dx + a3 * dx * dx * dx;

                    d0 = C[0] - C[1];
                    d2 = C[2] - C[1];
                    d3 = C[3] - C[1];
                    a0 = C[1];
                    a1 = -1.0 / 3 * d0 + d2 - 1.0 / 6 * d3;
                    a2 = 1.0 / 2 * d0 + 1.0 / 2 * d2;
                    a3 = -1.0 / 6 * d0 - 1.0 / 2 * d2 + 1.0 / 6 * d3;
                    Cc = a0 + a1 * dy + a2 * dy * dy + a3 * dy * dy * dy;

                    const uint8_t Cc2 = std::min(std::max(std::round(Cc), 0.0f), 255.0f);
                    dst.buf[(i * target_width + j) * 3 + k] = float(Cc2);
                }
            }
        }
    }

    return true;
}

void normalize_image_u8_to_f32(const clip_image_u8* src,
                                      clip_image_f32* dst,
                                      const float mean[3],
                                      const float std[3]) {
    dst->nx = src->nx;
    dst->ny = src->ny;
    dst->buf.resize(src->buf.size());

    for (size_t i = 0; i < src->buf.size(); ++i) {
        int c = i % 3;  // rgb
        dst->buf[i] = (static_cast<float>(src->buf[i]) / 255.0f - mean[c]) / std[c];
    }
}


void BlipModel::init(std::string blip_model_path, std::string device) {
    std::string vision_model_path = (std::filesystem::path(blip_model_path) / "blip_vision_model.xml").string();
    std::string projection_model_path = (std::filesystem::path(blip_model_path) / "blip_vision_proj_model.xml").string();
    try {
        vision_model = core.compile_model(vision_model_path, device).create_infer_request();
        std::cout << "Load blip vision model successed on " <<  device << " \n";
        vision_projection_model = core.compile_model(projection_model_path, device).create_infer_request();
        std::cout << "Load blip vision projection model successed\n";
        std::cout << "Init blip models successed\n";
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
    }
}

ov::Tensor BlipModel::preprocess(std::string image_path) {
    // load image
    clip_image_u8 image_u8;
    clip_image_f32 image_f32;

    if (!clip_image_load_from_file(image_path.c_str(), &image_u8)) {
        throw std::runtime_error("failed to load image");
    }

    // resize
    clip_image_u8 image_resized_u8;
    if (!bicubic_resize(image_u8, image_resized_u8, BLIP_PREPROCESS_DEFAULT_W, BLIP_PREPROCESS_DEFAULT_H)) {
        throw std::runtime_error("failed to resize image");
    }
    // normalize
    //normalize_image_u8_to_f32(&image_u8, &image_f32, OPENAI_CLIP_MEAN, OPENAI_CLIP_STD);
    
    normalize_image_u8_to_f32(&image_resized_u8, &image_f32, OPENAI_CLIP_MEAN, OPENAI_CLIP_STD);

    ov::Shape tensor_shape = {1,3, BLIP_PREPROCESS_DEFAULT_H, BLIP_PREPROCESS_DEFAULT_W};
    ov::Tensor image(ov::element::f32, tensor_shape);
    float* nchw_data = image.data<float>();
    // Convert NHWC(height, width, channels) to NCHW(channels, height, width)
    for (int h = 0; h < BLIP_PREPROCESS_DEFAULT_H; ++h) {
        for (int w = 0; w < BLIP_PREPROCESS_DEFAULT_W; ++w) {
            for (int c = 0; c < 3; ++c) {
                nchw_data[c * BLIP_PREPROCESS_DEFAULT_H * BLIP_PREPROCESS_DEFAULT_W + h * BLIP_PREPROCESS_DEFAULT_W +
                          w] = image_f32.buf[h * BLIP_PREPROCESS_DEFAULT_W * 3 + w * 3 + c];
            }
        }
    }
    return image;
}

std::vector<float> BlipModel::encode_image(std::string image_path) {
    try {
        ov::Tensor image = preprocess(image_path);

        vision_model.set_input_tensor(image);
        vision_model.infer();
        ov::Tensor pooled_output = vision_model.get_output_tensor(1);

        vision_projection_model.set_input_tensor(pooled_output);
        vision_projection_model.infer();
        ov::Tensor res = vision_projection_model.get_output_tensor(0);

        float* output_buffer = res.data<float>();
        auto shape = res.get_shape();
        std::cout << "DEBUG: embedding output shape: " << shape << std::endl;

        std::vector<float> embedding_result;

        for (size_t i = 0; i < shape[0]; i++) {
            for (size_t j = 0; j < shape[1]; j++) {
                embedding_result.push_back(output_buffer[i, j]);
            }
        }
        return embedding_result;
    } catch (const std::exception& ex) {
        std::cout << ex.what() << std::endl;
    }
}

std::vector<std::vector<float>> BlipModel::encode_images(std::vector<std::string> image_paths) {
    std::cout << "size of images: " << image_paths.size() << std::endl;
    std::cout << "Start Embedding " << std::endl;

    std::vector<std::vector<float>> embeddings;
    for (size_t i = 0; i < image_paths.size(); i++) {
        embeddings.push_back(encode_image(image_paths[i]));
    }
    std::cout << "shape of embedding_results: (" << embeddings.size() << ", " << embeddings[0].size()
              << ")" << std::endl;
    std::cout << "embedding infer successed\n";
    return embeddings;
}