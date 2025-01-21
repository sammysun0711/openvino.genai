// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/llm_pipeline.hpp"
#include <string>
#include <iostream>
#include <fstream>

#ifdef _WIN32
#include <Windows.h>
#include <io.h>
#else
#include <unistd.h>
#endif

int main(int argc, char* argv[]) try {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
    system("chcp 65001"); //Using UTF-8 Encoding
#endif
    if (3 != argc) {
        throw std::runtime_error(std::string{ "Usage: " } + argv[0] + " <MODEL_DIR>" + "<DEVICE>");
    }
    std::string prompt;
    std::string models_path = argv[1];


    //std::string device = "CPU";  // GPU, NPU can be used as well
    std::string device = argv[2];

    ov::genai::GenerationConfig config;
    config.max_new_tokens = 500;
    config.do_sample = false;
    config.top_k = 50;
    config.top_p = 0.1;
    config.stop_token_ids = { 59246, 59253,59255 };
    //config.temperature = 0.1;

    std::function<bool(std::string)> streamer = [](std::string word) {
        std::cout << word << std::flush;
        // Return flag corresponds whether generation should be stopped.
        // false means continue generation.
        return false;
        };

    std::cout << "question:\n";
    prompt = "Please summarize the document below, outputting the abstract and keywords:IEEE TRANSACTIONS ON INSTRUMENTATION AND MEASUREMENT, VOL. XX, NO. XX, XX 2022 1Joint Attention-Guided Feature Fusion Network forSaliency Detection of Surface DefectsXiaoheng Jiang, Feng Yan, Yang Lu, Ke Wang, Shuai GuoTianzhu Zhang, Yanwei Pang, Senior Member, IEEE, Jianwei Niu, Senior Member, IEEE, and Mingliang XuAbstractâ€”Surface defect inspection plays an important role inthe process of industrial manufacture and production. ThoughConvolutional Neural Network (CNN) based defect inspectionmethods have made huge leaps, they still confront a lot ofchallenges such as defect scale variation, complex background,low contrast, and so on. To address these issues, we proposea joint attention-guided feature fusion network (JAFFNet) forsaliency detection of surface defects based on the encoder-decodernetwork. JAFFNet mainly incorporates a joint attention-guidedfeature fusion (JAFF) module into decoding stages to adaptivelyfuse low-level and high-level features. The JAFF module learns toemphasize defect features and suppress background noise duringfeature fusion, which is beneficial for detecting low-contrastdefects. In addition, JAFFNet introduces a dense receptive field(DRF) module following the encoder to capture features with richcontext information, which helps detect defects of different scales.The JAFF module mainly utilizes a learned joint channel-spatialattention map provided by high-level semantic features to guidefeature fusion. The attention map makes the model pay moreattention to defect features. The DRF module utilizes a sequenceof multi-receptive-field (MRF) units with each taking as inputsall the preceding MRF feature maps and the original input. Theobtained DRF features capture rich context information with alarge range of receptive fields. Extensive experiments conductedon SD-saliency-900, Magnetic tile, and DAGM 2007 indicate thatour method achieves promising performance in comparison withother state-of-the-art methods. Meanwhile, our method reachesa real-time defect detection speed of 66 FPS.Index Termsâ€”Feature fusion, channel-spatial attention, densereceptive field, saliency detection, surface defects.This work was supported in part by National Key R&D Program of Chinaunder Grant 2021YFB3301504, in part by the National Natural Science Foun-dation of China under Grant 62172371, U21B2037, 62036010, 62102370,61903341, 62106232, in part by China Postdoctoral Science Foundation underGrant 2021TQ0301, and in part by Foundation for University Key Researchof Henan Province (21A520040, 21A520002), Hangzhou Innovation Institute,Beihang University (NO. 2020-Y4-A-020), and CAAI-Huawei MindSporeOpenFund. (Corresponding author: Jianwei Niu, Mingliang Xu .)Xiaoheng Jiang, Yang Lu, Ke Wang, Shuai Guo, and Mingliang Xu arewith the School of Computer and Artificial Intelligence, Zhengzhou Univer-sity, Engineering Research Center of Intelligent Swarm Systems, Ministryof Education, National Supercomputing Center in Zhengzhou, Zhengzhou450001, China (e-mail: jiangxiaoheng@zzu.edu.cn; ieylu@zzu.edu.cn; iek-wang@zzu.edu.cn; iesguo@zzu.edu.cn; iexumingliang@zzu.edu.cn).Feng Yan is with the School of Computer and Artificial Intelligence,Zhengzhou University, Zhengzhou 450001, China (ieyanfeng@163.com).Tianzhu Zhang is with the School of Information Science and Technology,University of Science and Technology of China, Hefei 230026, China (e-mail:tzzhang@ustc.edu.cn).Yanwei Pang is with the School of Electrical Automation and Infor-mation Engineering, Tianjin University, Tianjin 300072, China (e-mail:pyw@tju.edu.cn).Jianwei Niu is with the State Key Laboratory of Virtual Reality Technologyand Systems, School of Computer Science and Engineering, Beihang Univer-sity, Beijing 100191, China, and Hangzhou Innovation Institute of BeihangUniversity, Hangzhou 310051, China (e-mail: niujianwei@buaa.edu.cn).Fig. 1. Challenges of surface defect inspection. (a) and (b) defects withdifferent scales. (c) defects with low contrast. (a) and (d) interference factorsin the background. The defects and interference factors are represented by redand yellow rectangles, respectively.I. I NTRODUCTIONSURFACE defect inspection is a key task in the process ofindustrial production and is essential for product qualitycontrol. Compared with the manual defect inspection methods,computer vision based automatic defect inspection technolo-gies have become more popular in industrial production due totheir superior defect inspection performance with faster speedand higher accuracy.Traditional defect inspection approaches generally considerthe surface defect inspection as a texture analysis issue and ex-ploit several classic strategies such as texture filters [1], texturestatistics [2], [3], texture modeling [4], and texture structure[5]. These methods rely heavily on specific texture informationand work well when the defects are simple. However, the sur-face defects in real industrial scenes usually exhibit complexityand diversity in appearance and scale, which brings a hugechallenge for accurate defect inspection. Fig. 1 demonstratesseveral typical issues about surface defect inspection, includingscale variation, low contrast, and background interference. Thered and yellow rectangles in Fig. 1 represent the defects andbackground interference, respectively. Fig. 1 (a) and (b) showarXiv:2402.02797v1 [cs.CV] 5 Feb 2024IEEE TRANSACTIONS ON INSTRUMENTATION AND MEASUREMENT, VOL. XX, NO. XX, XX 2022 2that the defects vary largely in scale. Some surface defectsare very small and have less than 80 pixels in a 256 Ã— 256image, as shown in Fig. 1 (a). Fig. 1 (c) shows the lowcontrast between the defects and the background caused byinappropriate lighting conditions. Fig. 1 (a) and (d) show thatthere exist some interference factors in the background whichare very similar to the defects and are hard to discriminate.Recently, deep learning methods based on convolutionalneural network (CNN) have made great progress in manycomputer vision tasks such as image classification [6]â€“[8],object detection [9]â€“[11], image segmentation [12]â€“[14], andso on. These methods are designed for general objects and cannot directly generalize to surface defect inspection due to theabove mentioned challenges. To handle these challenges, re-searchers have designed CNN-based models that target surfacedefect inspection, such as the region-level defect inspectionmethods [15]â€“[19] and the pixel-level ones [20]â€“[24]. Amongthese works, the pixel-level methods can provide more detailedinformation about defects, such as boundary, shape, and size.Most of these methods adopt the encoder-decoder structure asthe basic backbone, in which the decoder can be regardedas the fusion process of high-level features from the toplayers and low-level features from the corresponding bottomlayers. The high-level features contain more abstract semanticinformation, while the low-level features contain more finedetails. The combination of the two-level features is beneficialto defect inspection. However, these methods still suffer froma certain amount of inspection errors when the defects showweak appearances, which are usually characterized by lowcontrast, small area, or subtle scratch. That is mainly becausethese methods like [21], [22], [24] simply adopt direct additionor concatenation operations to combine low-level and high-level features, in which the features related to defects are proneto be drowned by the background during feature fusion.To solve this problem, we present a joint attention-guidedfeature fusion (JAFF) module, which can adaptively reservefeatures of defects during feature fusion. JAFF first computesa channel-spatial attention map using the high-level featuresand then uses it to refine the corresponding low-level features.Finally, JAFF concatenates the refined low-level features andhigh-level features in channel dimension. The high-level fea-tures are used to generate the attention map based on thefact that they contain rich semantic information about thedefects. As a result, the obtained attention map can emphasizethe most meaningful low-level defect features and suppressbackground noise during feature fusion, resulting in robustdefect-focused features. In addition, context information isalso crucial in defect detection, especially for those defects ofvarious scales. As the scale of defects changes significantly, thesize of the receptive field should change accordingly. To handlethis problem, we present a dense receptive field (DRF) moduleto capture rich local context information with dense receptivefields, as well as global context information. DRF utilizesthe multi-receptive-field (MRF) units connected densely topromote the multi-scale representation ability of features.Based on the proposed JAFF module and DRF module,we develop the joint attention-guided feature fusion network(JAFFNet) for saliency detection of surface defects. In sum-mary, the main contributions are as follows:1) We develop a joint attention-guided feature fusion net-work (JAFFNet) for saliency detection of surface defectsby introducing two plug-and-play modules, which canachieve end-to-end defect detection.2) We present a joint attention-guided feature fusion (JAFF)module to effectively fuse high-level and low-levelfeatures. It is able to select valuable low-level defectfeatures and suppress background interference duringfeature fusion via the learned joint channel-spatial at-tention map.3) We present a dense receptive field (DRF) module tocapture context information with larger and denser scaleranges. It exploits rich context information by denselyconnecting a series of multi-receptive-field units and canhandle defects with various scales.4) The experiments on three publicly available surface de-fect datasets, including SD-saliency-900, DAGM 2007,and Magnetic Tile, demonstrate that the proposedmethod not only achieves promising defect detectionperformance but also reaches a real-time detection speedof 66 FPS.II. R ELATED WORKSA. Traditional defect inspection methodsMost traditional surface defect inspection methods are basedon texture analysis, which can be broadly classified intofour categories: filter-based, statistic-based, model-based, andstructure-based approaches. Specifically, the filter-based meth-ods analyze texture features through filters, such as Fouriertransform, Gabor transform [1], and Wavelet transform. Thestatistic-based methods analyze texture features through sta-tistical distribution characteristics of the image, such as gray-level co-occurrence matrix (GLCM) [2], local binary pattern(LBP) [3]. The model-based methods describe texture featuresthrough statistics of model parameters, such as the randomfield model, and fractal model [4]. The structure-based ap-proaches analyze texture features through texture primitivesand spatial placement rules, such as [5]. And these methodsare most customized for specific types of defects, with poorreusability and generalization ability. In addition, these meth-ods cannot effectively deal with complicated defects.B. Deep-learning-based defect inspection methodsCompared with traditional surface defect inspection meth-ods, deep-learning-based defect inspection methods are ableto handle defects with weak characteristics and complexbackground, and show superiority in complex scenes. And webroadly divide these methods into two categories: region-leveland pixel-level defect inspection methods.1) Region-level inspection methods: These methods locatedefects by bounding boxes. To improve the defect detectionability of the model, He et al. [15] first integrate multi-levelfeatures into one feature and then feed it into the regionproposal network to generate high-quality defect proposals.Wei et al. [16] incorporate the attention-related visual gainIEEE TRANSACTIONS ON INSTRUMENTATION AND MEASUREMENT, VOL. XX, NO. XX, XX 2022 3Fig. 2. Architecture of the proposed network. Our model consists of an encoder and a decoder, where we obtain multi-level features with channels 64, 128,256, 512, and 512 from five encoding stages E1 âˆ¼ E5, respectively. And D1 âˆ¼ D4 represent four decoding stages with each including a joint attention-guidedfeature fusion (JAFF) module and a convolution block. And the JAFF focuses on the fusion of high-level and low-level features. It incorporates a dual attentionmodule consisting of a channel attention branch (CAB) and a spatial attention branch (SAB) to generate the learned channel-spatial attention map that providesguidance for feature fusion. The dense receptive field (DRF) module after the encoder is used to capture dense context information. And the â€œDSâ€ and â€œrâ€denote depthwise separable convolution and rate of dilated convolution, respectively.mechanism into the Faster RCNN model to improve thediscrimination ability of small defects. However, these detec-tors obtain bounding boxes based on region proposals, withhigh accuracy but slow speed. Therefore, Cui et al. [17]design a fast and accurate detector called SDDNet, whichdetects small defects by passing fine-grained details of low-level features to all deep features. Su et al. [18] adopt anovel attention module (RCAG) to fuse multi-scale features,with the aim of emphasizing defect features and suppressingbackground noise. Different from [17] and [18], Tu et al. [19]achieve accurate defect detection by adopting CIoU loss andintroducing the Gaussian function to estimate the coordinatesof the prediction boxes.2) Pixel-level inspection methods: These methods can pro-vide more structural details of defects than region-level meth-ods, such as boundary, shape, and size. It is essential foraccurate defect detection to capture and integrate multiplecontext information effectively. To this end, Huang et al.[20] apply an atrous spatial pyramid pooling (ASPP) in theproposed lightweight defect segmentation network to capturemultiple context information. Zhang et al. [21] integrate mul-tiple context information through the pyramid pooling module(PPM) and attention module, with the aim of enhancing defectfeatures and filtering out noise. Li et al. [22] integrate multi-scale features from encoder blocks step-by-step, which se-quentially fuses two adjacent scale features and three adjacentscale features. In addition, the attention mechanism is alsooften used in defect segmentation to address those defectswith complex background. For example, Song et al. [23]incorporate the attention mechanism into the model to steerit to focus more on defect features. Zhou et al. [24] introducedense attention cues into the decoder to make it more defect-focused.Different from these existing methods, we propose a jointattention-guided feature fusion network for saliency detectionof surface defects. Specifically, we design two modules toimprove the defect detection performance of encoder-decoderarchitecture. One is called JAFF module, which is used ateach decoding stage to retain more defect features during thefusion of low-level and high-level features. The other is calledDRF module, which is embedded after the fifth encodingstage to capture dense context information and strengthen therepresentation ability of deep features. And the proposed twomodules greatly improve defect detection performance of thenetwork in complex scenes.C. Attention Mechanism in CNNsThe attention mechanism can selectively focus on importantinformation while ignoring less useful information, which isimportant for understanding complex scenes. Hu et al. [25]first propose channel attention and perform adaptive featureIEEE TRANSACTIONS ON INSTRUMENTATION AND MEASUREMENT, VOL. XX, NO. XX, XX 2022 4recalibration by explicitly modeling global information. Dueto the limitation of channel attention, Woo et al. [26] proposethe convolutional block attention module (CBAM) whichapplies both channel-wise and spatial attention in sequential.CBAM not only introduces spatial attention but also introducesboth max-pooled and average-pooled features in the spatialaxis into channel attention. Park et al. [27] also design the";

    /*
    if (_putenv("ONEDNN_PRIMITIVE_CACHE_CAPACITY=0") != 0) {
        throw std::runtime_error("Failed to set environment variable");
    }
    */
    
    // ov::genai::LLMPipeline* pipe = new ov::genai::LLMPipeline(models_path, device);

    for (int i = 0; i < 3; ++i) {
        std::cout << "Round: " << i << "\n";
        ov::genai::LLMPipeline* pipe = new ov::genai::LLMPipeline(models_path, device);
        std::cout << "Init pipeline 1 works\n";
        /*
        ov::genai::LLMPipeline* pipe1 = new ov::genai::LLMPipeline(models_path, device);
        std::cout << "Init pipeline 2 works\n";
        */
        std::string prompt_utf8 = prompt;

        {
            auto start = std::chrono::high_resolution_clock::now();
            pipe->start_chat();
            std::string result = pipe->generate(prompt_utf8, config, streamer);
            //std::string result = pipe->generate(StringToUTF8(prompt), config);
            std::cout << result << "result\n";
            pipe->finish_chat();
            auto end1 = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> dur1 = end1 - start;
            std::cout << "\nE2E Inference time takes: " << dur1.count() << " ms" << std::endl;
        }
        std::cout << "Delete pipe called " << std::endl;
        delete pipe;
        //delete pipe1;

        std::cout << "ov::genai::clear_core_device(" << device << ")\n";
        ov::genai::clear_core_device(device);
        std::cout << "After delete pipe & unload plugin " << std::endl;
	#ifdef _WIN32
	    Sleep(20000);
	#else
        sleep(20);
	#endif
    }
}
catch (const std::exception& error) {
    try {
        std::cerr << error.what() << '\n';
    }
    catch (const std::ios_base::failure&) {}
    return EXIT_FAILURE;
}
catch (...) {
    try {
        std::cerr << "Non-exception object thrown\n";
    }
    catch (const std::ios_base::failure&) {}
    return EXIT_FAILURE;
}

