#include "blip.hpp"

int main () {
    BlipModel model;
    std::string model_path = "C:\\Users\\chuxd\\repos\\ovgenai_rag\\samples\\cpp\\rag_sample\\blip_vqa_base";
    model.init(model_path, "CPU");

    std::string image_path = "C:\\Users\\chuxd\\repos\\ovgenai_rag\\samples\\cpp\\rag_sample\\scripts\\demo.png";
    std::vector<std::string> paths = {image_path};
    auto embeddings = model.encode_images(paths);
    return 0;
}