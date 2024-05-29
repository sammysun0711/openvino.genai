#include "qwen.h"
#include "sampling.hpp"

#include <regex>
#include <random>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <regex>
#include <vector>

#include <openvino/openvino.hpp>
#include <openvino/runtime/properties.hpp>
#include "openvino/runtime/intel_gpu/properties.hpp"
#include "openvino/pass/serialize.hpp"
#include "openvino/op/ops.hpp"
#include "openvino/opsets/opset13.hpp"
#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/pass/serialize.hpp"

#ifdef _WIN32
#include <codecvt>
#include <fcntl.h>
#include <io.h>
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

typedef std::chrono::high_resolution_clock Time;
typedef std::chrono::nanoseconds ns;

/*@brief Insert slice transformation matches following graph, start from logits (Results) to search along root->parent-> grandparent node,
 * then insert slice between Reshape (grandparent node) and Matmul to keep only last dim of matmul first input, first input shape reduced
 * from [1, seq_len, 4096] to [1, 1,4096]. Therefore, after graph transformation, we can reduce matmul computation
 * from [1, seq_len, 4096] * [1, 4096, 151936] = [1, seq_len, 151936] to [1,1,4096]*[4096,151936] = [1,1,151936]
 *
 * Original graph
 *         +----------+            +----------+
 *         |  Reshape |            | Constant |
 *         +----------+            +----------+
 *              |                       |
 *              -----------    ----------
 *                        |    |
 *                        v    v
 *                      +--------+
 *                      | MatMul |
 *                      +--------+
 *                          |
 *                          v
 *                     +----------+
 *                     |  logits  |
 *                     +----------+
 *
 * Modified graph after insert slice:
 *f
 *         +----------+            +----------+
 *         |  Reshape |            | Constant |
 *         +----------+            +----------+
 *              |                       |
 *         +----------+                 |
 *         |  Slice   |                 |
 *         +----------+                 |
 *              |                       |
 *              -----------    ----------
 *                        |    |
 *                        v    v
 *                      +--------+
 *                      | MatMul |
 *                      +--------+
 *                          |
 *                          v
 *                     +----------+
 *                     |  logits  |
 *                     +----------+
 */

class InsertSlice : public ov::pass::MatcherPass
{
public:
  OPENVINO_RTTI("InsertSlice", "0");
  explicit InsertSlice()
  {
    auto label = ov::pass::pattern::wrap_type<ov::op::v0::Result>();
    ov::matcher_pass_callback callback = [=](ov::pass::pattern::Matcher &m)
    {
      auto root = std::dynamic_pointer_cast<ov::op::v0::Result>(m.get_match_root());
      if (!root)
      {
        return false;
      }
      std::string root_name = root->get_friendly_name();
      if (root->get_output_partial_shape(0).size() == 3)
      {
        std::cout << "Find target root node name: " << root_name << "\n";
        auto parent = root->input_value(0).get_node_shared_ptr();
        std::cout << "Find parent node name: " << parent->get_friendly_name() << "\n";
        auto grand_parent = parent->input_value(0).get_node_shared_ptr();
        std::cout << "Find grandparent node name: " << grand_parent->get_friendly_name() << "\n";
        ov::Output<ov::Node> grand_parent_output = parent->get_input_source_output(0);
        std::set<ov::Input<ov::Node>> consumers = grand_parent_output.get_target_inputs();
        std::vector<int32_t> start_v = {0, -1, 0};
        std::vector<int32_t> stop_v = {1, -2, 4096};
        std::vector<int32_t> step_v = {1, -1, 1};
        std::cout << "Original reshape node output shape:" << grand_parent_output.get_partial_shape() << std::endl;
        auto starts = ov::op::v0::Constant::create(ov::element::i32,
                                                   ov::Shape{3},
                                                   start_v);
        auto stop = ov::op::v0::Constant::create(ov::element::i32,
                                                 ov::Shape{3},
                                                 stop_v);
        auto step = ov::op::v0::Constant::create(ov::element::i32,
                                                 ov::Shape{3},
                                                 step_v);
        auto slice = std::make_shared<ov::opset13::Slice>(grand_parent, starts, stop, step); // data, starts, ends, steps
        std::cout << "After insert slice node, output shape" << slice->output(0).get_partial_shape() << std::endl;
        for (auto consumer : consumers)
        {
          consumer.replace_source_output(slice->output(0));
        }
        register_new_node(slice);
      }

      return true;
    };
    // Register pattern with Parameter operation as a pattern root node
    auto m = std::make_shared<ov::pass::pattern::Matcher>(label, "InsertSlice");
    // Register Matcher
    register_matcher(m, callback);
  }
};
const std::vector<std::string> english_sentences =
    {
        "What is OpenVINO?",
        "If I have 100 million dollars, what kinds of projects should I invest to maximize my benefits in background of a growing number of artificial intelligence technologies?",
        "Originally, There were three types of cake in the cake store: Strawberry Cream Cake, Chocolate Coconut Cake, and Red Velvet Brownie Cake. Customer number is large enough so that no cake would be left every day when the store close. As the name suggested, each cake has two ingredients: Strawberry Cream Cake with strawberries and cream, Chocolate Coconut Cake with chocolate and coconut, and Red Velvet Brownie Cake with red velvet and brownie. Different ingredients can be compatibly mixed with each other without any issue. After the cake is made, there are often some leftover materials for each ingredient. In order to reduce waste, the store often combine the extra ingredients in pairs to make new small gifts to gain extra sales. For example, strawberries and chocolate can be mixed to create strawberry-flavored chocolate sauce, and brownies and shredded coconut can be mixed to create brownie coconut cookies. Only two ingredients can be mixed, and mixture with more than two ingredients can cost a lot of time and will not be adopted. In order to decrease the problem complexity, the store will also prevent from careful decorations or other difficult steps as in procedure of making cakes, so that time cost can be omited. By analogy, if all the ingredients can be combined in pairs, what small products can the store make in the end?",
        "There is a table, which contains three drawers: left drawer, middle drawer and right drawer; Tom Ethan, Elbert Alex, Jack Johnson, and Mario Thompson all saw a bag of chocolates on the table. Tom Ethan asked Elbert Alex and Jack Johnson to go out, and after that, he put the bag of chocolates in the right drawer in front of Mario Thompson; after Jack Johnson came back, Tom Ethan asked Mario Thompson to go out to find Elbert Alex, and took it from the left drawer in front of Jack Johnson. Then He take out a box of biscuits and put them in the middle drawer; when Elbert Alex and Mario Thompson returned, Tom Ethan asked Jack Johnson and Mario Thompson to go out to buy a bottle of soy sauce. Tom Ethan waited for a long time, and found that Jack Johnson and Mario Thompson had not returned, so he sent Elbert Alex to look for them, but in the end only Jack Johnson and Elbert Alex came back. Jack Johnson told Tom Ethan that at first they could not find any shop that is providing soy sauce, so they had to separate to search other shops, which is why Mario Thompson got lost; on the way back, Jack Johnson ran into Elbert Alex, and they rushed back first. Therefore, Tom Ethan asked them to go out to find Mario Thompson again; in order to prevent getting lost again, Tom Ethan told Elbert Alex and Jack Johnson to walk together at all time, and even if they could not get the soy sauce, they had to find and get back with Mario Thompson. As a result, Elbert Alex and Jack Johnson found Mario Thompson outside and found that he had bought a bottle of soy sauce. The three felt that Tom Ethan never went out to do anthing but they are busy all the time. So they were very angry. They discussed and made a conclusion. After going back to see Tom Ethan, they should not tell him about the soy sauce they bought, and asked Jack Johnson to hide the soy sauce in his backpack. After the three of them came back together, they pretended to claim that they did not foudn and bought soy sauce according to the plan, and hoped that Tom Ethan would go out together to buy things in the future, and he should not be so lazy. Tom Ethan agreed and felt sory about that. When everyone finally stood in front of the table, the four of them wrote down the list of items they knew and the location of the items. So the question is: is the information writen by these four people consistent, and why?",
        "The process of Origami seems simple at the first glance, but in fact, it still requires a very complicated process to do it well. Taking folding a rose as an example, we can divide the entire process into three stages, including: firstly creating a grid of creases, secondly making a three-dimensional base, and thirdly finishing petal decoration. The first step is to create a grid of creases: this step is a bit like the first step of folding a gift of thousand-paper-crane. That is to say, we can fold the paper in half (or namedly equal-folds) through the symmetrical axis, and repeat such step in the other symmetrical axis. And then apply multiple equal-folds in sequence relative to each smaller rectangle divided by the two creases; After that, the creases in each direction will interweave into a complete set of uniform small square splicing patterns; these small squares form a reference space similar to a two-dimensional coordinate system, allowing us to combine adjacent creases on the plane from Three-dimensional high platforms or depressions are folded on the two-dimensional small squares to facilitate the next steps of folding. It should be noted that, in the process of creating grid creases, there may be rare cases when the folds are not aligned. The consequences of this error can be very serious. And just like the butterfly effect, it is only a slight difference at the beginning , and in the end it may generate a disaster world which is completely different from plan. Anyway, let's continue. The second step is make the three-dimensional base: In this step, we need to fold a set of symmetrical three-dimensional high platforms or depressions based on the grid creases. From the symmetry analysis, it is not difficult to find that the rose will have four symmetrical three-dimensional high platforms and supporting depressions. Therefore, we can firstly fold out a quarter of the depression and plateau patterns, which would help build a base to compose into a complex 3D structure. And then, we use this quarter as a template, and fold out the repeating patterns on the remaining three parts of the whole structure in turn. It is worth noting that the layout of the high platform not only needs to consider the regular contrast and symmetrical distribution of the length and width, but also needs to ensure the orderliness of the height dimension. This is very important, since we will never go back to this step after all parts were made, and you would better start from first step if you make anything wrong in the this step. Similar to the precautions in the first stage, please handle all the corners in three dimensions to ensure that they conform to the layout required in the plan, which would help us avoid the butterfly effect and increase the robustness in the process of three-dimensional folding. Just like building a skyscrapper in the real world, people usually take a lot of time when building the base but soon get finished when extending the structure after that. Time is worth to cost in the base, but would be saved in the future after you succeed in base. Anyway, let's continue. During the first quarter of the pattern, repeated comparisons with the finished rose were made to eliminate any possible errors in the first place. The final stage is to finish the petal grooming. At this stage, we often emphasize an important term called folding-by-heart. The intention here is no longer literally serious, but focus is moved to our understanding of the shape of a rose in nature, and we usually use natural curves to continuously correct the shape of petals in order to approach the shape of rose petals in reality. One more comment: this is also the cause of randomness to the art, which can be generated differently by different people. Recall that rose should be adjusted close to reality, so in the last step of this stage, we need to open the bloom in the center of the rose, by pulling on the four petals that have been bent. This process may be accompanied by the collapse of the overall structure of the rose, so we should be very careful to save strength of adjustment, and it must be well controlled to avoid irreversible consequences. Ultimately, after three stages of folding, we end up with a crown of rose with a similar shape close to reality. If condition is permited, we can wrap a green paper strip twisted on a straightened iron wire, and insert the rose crown we just created onto one side of the iron wire. In this way, we got a hand-made rose with a green stem. We can also repeat the steps above to increase the number of rose, so that it can be made into a cluster. Different color of rose is usually more attractive and can be considered as a better plan of gift to your friend. In summary, by creating a grid of creases, making a three-dimensional base, and finishing with petals, we created a three-dimensional rose from a two-dimensional paper. Although this process may seem simple, it is indeed a work of art created by us humans with the help of imagination and common materials. At last, Please comment to assess the above content.",
        "患者男，年龄29岁，血型O，因思维迟钝，易激怒，因发热伴牙龈出血14天，乏力、头晕5天就诊我院急诊科。快速完善检查，血常规显示患者三系血细胞重度减低，凝血功能检查提示APTT明显延长，纤维蛋白原降低，血液科会诊后发现患者高热、牙龈持续出血，胸骨压痛阳性.于3903年3月7日入院治疗，出现头痛、头晕、伴发热（最高体温42℃）症状，曾到其他医院就医。8日症状有所好转，9日仍有头痛、呕吐，四肢乏力伴发热。10日凌晨到本院就诊。患者5d前出现突发性思维迟钝，脾气暴躁，略有不顺心就出现攻击行为，在院外未行任何诊治。既往身体健康，平素性格内向。体格检查无异常。血常规白细胞中单核细胞百分比升高。D-二聚体定量1412μg/L，骨髓穿刺示增生极度活跃，异常早幼粒细胞占94%.外周血涂片见大量早幼粒细胞，并可在胞浆见到柴捆样细胞.以下是血常规详细信息：1.病人红细胞计数结果：3.2 x10^12/L. 附正常参考范围：新生儿:（6.0～7.0）×10^12/L；婴儿：（5.2～7.0）×10^12/L; 儿童：（4.2～5.2）×10^12/L; 成人男：（4.0～5.5）×10^12/L; 成人女：（3.5～5.0）×10^12/L. 临床意义：生理性红细胞和血红蛋白增多的原因：精神因素（冲动、兴奋、恐惧、冷水浴刺激等导致肾上腺素分泌增多的因素）、红细胞代偿性增生（长期低气压、缺氧刺激，多次献血）；生理性红细胞和血红蛋白减少的原因：造血原料相对不足，多见于妊娠、6个月～2岁婴幼儿、某些老年性造血功能减退；病理性增多：多见于频繁呕吐、出汗过多、大面积烧伤、血液浓缩，慢性肺心病、肺气肿、高原病、肿瘤以及真性红细胞增多症等；病理性减少：多见于白血病等血液系统疾病；急性大出血、严重的组织损伤及血细胞的破坏等；合成障碍，见于缺铁、维生素B12缺乏等。2. 病人血红蛋白测量结果：108g/L. 附血红蛋白正常参考范围：男性120～160g/L；女性110～150g/L；新生儿170～200g/L；临床意义：临床意义与红细胞计数相仿，但能更好地反映贫血程度，极重度贫血（Hb<30g/L）、重度贫血（31～60g/L）、中度贫血（61～90g/L）、男性轻度贫血（90~120g/L）、女性轻度贫血（90~110g/L）。3. 病人白细胞计数结果：13.6 x 10^9/L; 附白细胞计数正常参考范围：成人（4.0～10.0）×10^9/L；新生儿（11.0～12.0）×10^9/L。临床意义：1）生理性白细胞计数增高见于剧烈运动、妊娠、新生儿；2）病理性白细胞增高见于急性化脓性感染、尿毒症、白血病、组织损伤、急性出血等；3）病理性白细胞减少见于再生障碍性贫血、某些传染病、肝硬化、脾功能亢进、放疗化疗等。4. 病人白细胞分类技术结果：中性粒细胞（N）50%、嗜酸性粒细胞（E）3.8%、嗜碱性粒细胞（B）0.2%、淋巴细胞（L）45%、单核细胞（M）1%。附白细胞分类计数正常参考范围：中性粒细胞（N）50%～70%、嗜酸性粒细胞（E）1%～5%、嗜碱性粒细胞（B）0～1%、淋巴细胞（L）20%～40%、单核细胞（M）3%～8%；临床意义：1）中性粒细胞为血液中的主要吞噬细胞，在细菌性感染中起重要作用。2）嗜酸性粒细胞①减少见于伤寒、副伤寒、大手术后、严重烧伤、长期用肾上腺皮质激素等。②增多见于过敏性疾病、皮肤病、寄生虫病，一些血液病及肿瘤，如慢性粒细胞性白血病、鼻咽癌、肺癌以及宫颈癌等；3）嗜碱性粒细胞 a 减少见于速发型过敏反应如过敏性休克，肾上腺皮质激素使用过量等。b 增多见于血液病如慢性粒细胞白血病，创伤及中毒，恶性肿瘤，过敏性疾病等；4）淋巴细胞 a 减少多见于传染病的急性期、放射病、细胞免疫缺陷病、长期应用肾上腺皮质激素后或放射线接触等。b 增多见于传染性淋巴细胞增多症、结核病、疟疾、慢性淋巴细胞白血病、百日咳、某些病毒感染等；5）单核细胞增多见于传染病或寄生虫病、结核病活动期、单核细胞白血病、疟疾等。5. 病人血小板计数结果：91 x10^9/L. 附血小板计数正常参考范围：（100～300）×10^9/L. 临床意义：1）血小板计数增高见于真性红细胞增多症、出血性血小板增多症、多发性骨髓瘤、慢性粒细胞性白血病及某些恶性肿瘤的早期等；2）血小板计数减低见于 a 骨髓造血功能受损，如再生障碍性贫血，急性白血病；b 血小板破坏过多，如脾功能亢进；c 血小板消耗过多，如弥散性血管内凝血等。6. 以往病例分析内容参考：白血病一般分为急性白血病和慢性白血病。1）急性白血病血常规报告表现为：白细胞增高，少数大于100×10^9/L，称为高白细胞白血病，部分患者白细胞正常或减少，低者可小于1.0×10^9/L，以AML中的M3型多见。在白细胞分类中，80％以上可见大量的幼稚细胞，有时仅见幼稚细胞和少量成熟的细胞，而无中间型细胞，称为白血病的裂孔现象。少数白细胞低的患者周围血幼稚细胞很少，此类患者必须骨髓穿刺才能确诊。多数急性白血病患者初诊时有不同程度的贫血；一般属正常细胞正色素型。但贫血很快会进行性加重。30％的患者血涂片中可见有核红细胞。血小板计数绝大部分患者减少，严重者小于10×10^9/L，仅极少数患者血小板计数正常。2） 慢性白血病血常规报告表现为：白细胞总数明显增高，通常大于30×10^9/L。半数患者大于100×10^9/L。中性粒细胞明显增多，可见各阶段粒细胞，以中性中幼粒，晚幼粒细胞居多，原始粒细胞小于等于10％，通常为1％～5％，嗜酸和嗜碱粒细胞亦增多。初诊时约有50％患者血小板增高，少数可大于1000×10^9/L。红细胞和血红蛋白一般在正常范围，若出现血红蛋白减低，血小板计数明显升高或降低，则提示疾病向加速期或急变期转化。7. 历史相关研究: 急性髓系白血病（AML）是造血干细胞恶性克隆性疾病。在AML的诊断、治疗以及判断预后的过程中，基因异常是一项重要指标。随着基因检测技术的不断进步，越来越多与AML发生相关的基因被人们发现，并且这些基因在指导预后方面有重要意义。常见和急性髓系白血病相关的基因突变有: 1）RUNX1-RUNX1T1; 2）CBFB-MYH11; 3）NPM1：核磷蛋白1（nucleophosmin 1，NPM1）; 4）CEBPA：CCAAT增强子结合蛋白α基因（CCAAT/en－hancer binding protein α，CEBPA）; 5）MLLT3-KMT2A; 6）DEK-NUP214; 7）KMT2A：KMT2A基因（也称为MLL基因）问：请基于以上信息做出判断，该患者是否有罹患急性白血病的风险？请结合上述内容给出判断的详细解释，并简要总结潜在的早期征兆、预防方法、相关的基因突变、常用的治疗手段，以及当前已上市和正在临床阶段的药物清单。答：",
        "Originally, There were three types of cake in the cake store: Strawberry Cream Cake, Chocolate Coconut Cake, and Red Velvet Brownie Cake. Customer number is large enough so that no cake would be left every day when the store close. As the name suggested, each cake has two ingredients: Strawberry Cream Cake with strawberries and cream, Chocolate Coconut Cake with chocolate and coconut, and Red Velvet Brownie Cake with red velvet and brownie. Different ingredients can be compatibly mixed with each other without any issue. After the cake is made, there are often some leftover materials for each ingredient. In order to reduce waste, the store often combine the extra ingredients in pairs to make new small gifts to gain extra sales. For example, strawberries and chocolate can be mixed to create strawberry-flavored chocolate sauce, and brownies and shredded coconut can be mixed to create brownie coconut cookies. Only two ingredients can be mixed, and mixture with more than two ingredients can cost a lot of time and will not be adopted. In order to decrease the problem complexity, the store will also prevent from careful decorations or other difficult steps as in procedure of making cakes, so that time cost can be omited. By analogy, if all the ingredients can be combined in pairs, what small products can the store make in the end?",
        "If I have 100 million dollars, what kinds of projects should I invest to maximize my benefits in background of a growing number of artificial intelligence technologies?",
};

const std::vector<std::string> chinese_sentences =
    {
        "介绍下清华大学",
        "若我有一亿美元，在人工智能盛行的今天，我怎样投资才能收益最大化？",
        "糕点商店里原本有三种蛋糕：草莓奶油蛋糕，巧克力椰蓉蛋糕，和红丝绒布朗尼蛋糕。如名字所描述的那样，每种蛋糕都有两种成分：草莓奶油蛋糕包含草莓和奶油两个成分，巧克力椰蓉蛋糕包含巧克力和椰蓉两种成分，红丝绒布朗尼蛋糕包含红丝绒和布朗尼两种成分。在蛋糕制作完成后，往往每一种成分的材料都会有所剩余。为了减少浪费，商店常常会把多出来的成分两两搭配，做成新的小商品卖出去。比如草莓和巧克力可以做成草莓味巧克力酱，布朗尼和椰蓉可以做成布朗尼椰蓉饼干。以此类推可知，如果所有的成分都可以两两组合，那么最终商店能做出哪些小商品出来？",
        "桌子有左中右3个抽屉；张三，李四，王五，赵六都看到桌子上有一袋巧克力。张三让李四和王五出门后，在赵六面前把这袋巧克力放进了右抽屉；王五回来后，张三让赵六出门去找李四，并在王五面前从左抽屉拿出一盒饼干放进中抽屉里；等李四和赵六返回，张三又让王五和赵六出去买酱油，等二人走后，他告诉李四刚才已将一盒饼干放进中抽屉；张三等了很久，发现王五和赵六还没回来，就派李四去寻找，可最后只有王五和李四回来了。王五告诉张三，一开始他们没有找到卖酱油的店，所以只好分头去买，后来赵六走丢了；回来的路上，王五碰上了李四，两人便先赶了回来。于是，张三让两人出门去找赵六；为防再次走丢，张三叮嘱李四和王五要时刻同行，就算酱油买不到，也要找回赵六。结果，李四和王五在外面找到了赵六，发现他已经买了酱油。三人觉得张三从来不出门跑腿，十分气愤，讨论并达成共识，回去见到张三后，不要告诉他买到了酱油的事情，并让王五把酱油藏到自己的背包里。等三人一同回来后，他们按照计划谎称没有买到酱油，并希望张三以后买东西也要一同出门，不能偷懒，张三答应了。当大家最后站在桌子前，四人分别写下自己知道的物品清单和物品所在位置。问，这四人写下的物品和位置信息是否一致，为什么？",
        "折纸的过程看似简单，其实想要做好，还是需要一套很复杂的工艺。以折一支玫瑰花为例，我们可以将整个折纸过程分成三个阶段，即：创建栅格折痕，制作立体基座，完成花瓣修饰。首先是创建栅格折痕：这一步有点像我们折千纸鹤的第一步，即通过对称州依次对折，然后按照长和宽两个维度，依次进行多等分的均匀折叠；最终在两个方向上的折痕会交织成一套完整均匀的小方格拼接图案；这些小方格就组成了类似二维坐标系的参考系统，使得我们在该平面上，通过组合临近折痕的方式从二维小方格上折叠出三维的高台或凹陷，以便于接下来的几座制作过程。需要注意的是，在建立栅格折痕的过程中，可能会出现折叠不对成的情况，这种错误所带来的后果可能是很严重的，就像是蝴蝶效应，一开始只是毫厘之差，最后可能就是天壤之别。然后是制作立体基座：在这一步，我们需要基于栅格折痕折出对称的三维高台或凹陷。从对称性分析不难发现，玫瑰花会有四个周对称的三维高台和配套凹陷。所以，我们可以先折出四分之一的凹陷和高台图案，然后以这四分之一的部分作为摸板，再依次折出其余三个部分的重复图案。值得注意的是，高台的布局不仅要考虑长和宽这两个唯独上的规整衬度和对称分布，还需要同时保证高这个维度上的整齐。与第一阶段的注意事项类似，请处理好三个维度上的所有折角，确保它们符合计划中所要求的那种布局，以免出现三维折叠过程中的蝴蝶效应；为此，我们常常会在折叠第一个四分之一图案的过程中，与成品玫瑰花进行反复比较，以便在第一时间排除掉所有可能的错误。最后一个阶段是完成花瓣修饰。在这个阶段，我们往往强调一个重要名词，叫用心折叠。这里的用心已经不是字面上的认真这个意思，而是指通过我们对于大自然中玫瑰花外型的理解，借助自然的曲线去不断修正花瓣的形状，以期逼近现实中的玫瑰花瓣外形。请注意，在这个阶段的最后一步，我们需要通过拉扯已经弯折的四个花瓣，来调整玫瑰花中心的绽放程度。这个过程可能会伴随玫瑰花整体结构的崩塌，所以，一定要控制好调整的力道，以免出现不可逆的后果。最终，经过三个阶段的折叠，我们会得到一支栩栩如生的玫瑰花冠。如果条件允许，我们可以在一根拉直的铁丝上缠绕绿色纸条，并将玫瑰花冠插在铁丝的一段。这样，我们就得到了一支手工玫瑰花。总之，通过创建栅格折痕，制作立体基座，以及完成花瓣修饰，我们从二维的纸面上创作出了一支三维的花朵。这个过程虽然看似简单，但它确实我们人类借助想象力和常见素材而创作出的艺术品。请赏析以上内容的精妙之处。",
};

double get_duration_ms_until_now(Time::time_point &startTime)
{
  return std::chrono::duration_cast<ns>(Time::now() - startTime).count() * 0.000001;
}

struct Args
{
  std::string model_path = "openvino_model.xml";
  std::string tiktoken_path = "qwen.tiktoken";
  std::string prompt = "";
  int max_length = 2048;
  int max_context_length = 256;
  std::string device = "CPU";
  bool verbose = false;
  std::string language = "english";
  bool reduce_logits = false;
  int num_iteration = 1;
  bool force_max_generation = 0;
  bool do_sample = false;
  int top_k = 0;
  float top_p = 0.7;
  float temp = 0.95;
  float repeat_penalty = 1.0;
  int repeat_last_n = 64;
  int seed = 0;
  bool enable_latency_log = false;
};

static auto usage(const std::string &prog) -> void
{
  std::cout << "Usage: " << prog << " [options]\n"
            << "\n"
            << "options:\n"
            << "  -h,    --help                               Show this help message and exit\n"
            << "  -m,    --model                  PATH        Model path (default: openvino_model.xml)\n"
            << "  -t,    --tiktoken_path          PATH        Tokenizer path (default: qwen.tiktoken)\n"
            << "  -p,    --prompt                 STRING      Input prompt from user\n"
            << "  -ml,   --max_length             N           Max total length including prompt and output (default: 2048)\n"
            << "  -mcl,  --max_context_length     N           Max context length (default: 256)\n"
            << "  -d,    --device                 STRING      Specify which device used for inference\n"
            << "  -l,    --language               STRING      Specify test sentence language, either english or chinese\n"
            << "  -ds,   --do_sample              BOOL        Specify whether do random sample (default: False)\n"
            << "  -tk,   --top_k                  N           Specify top-k parameter for sampling (default: 0)\n"
            << "  -tp,   --top_p                  N           Specify top-p parameter for sampling (default: 0.7)\n"
            << "  -temp, --temperature            N           Specify temperature parameter for sampling (default: 0.95)\n"
            << "  -sd,   --seed                   N           Specify seed for sampling (default: 0)\n"
            << "  -rp,   --repeat_penalty         N           Specify penalize sequence of tokens (default: 1.0, means no repeat penalty)\n"
            << "  -rln,  --repeat_last_n          N           Specify last N tokens for penalize (default: 64)\n"
            << "  -r,    --reduce_logits          N           Apply graph optimization to reduce logits calculation of last matmul (default: 0)\n"
            << "  -n     --num_iteration          N           Specify how many iteration used for text sentence, (default: 1)\n"
            << "  -f     --force_max_generation   BOOL        Force llm to generate to max_context_length, (default: 0)\n"
            << "  -v,    --verbose                BOOL        Display verbose output including config/system/performance info\n"
            << "  -ell   --enable_latency_log     BOOL        Enable to save second latency per sentence in disk\n";
}

static auto parse_args(const std::vector<std::string> &argv) -> Args
{
  Args args;

  for (size_t i = 1; i < argv.size(); i++)
  {
    const std::string &arg = argv[i];

    if (arg == "-h" || arg == "--help")
    {
      usage(argv[0]);
      exit(EXIT_SUCCESS);
    }
    else if (arg == "-m" || arg == "--model")
    {
      args.model_path = argv[++i];
    }
    else if (arg == "-t" || arg == "--tiktoken_path")
    {
      args.tiktoken_path = argv[++i];
    }
    else if (arg == "-p" || arg == "--prompt")
    {
      args.prompt = argv[++i];
    }
    else if (arg == "-ml" || arg == "--max_length")
    {
      args.max_length = std::stoi(argv[++i]);
    }
    else if (arg == "-mcl" || arg == "--max_context_length")
    {
      args.max_context_length = std::stoi(argv[++i]);
    }
    else if (arg == "-d" || arg == "--device")
    {
      args.device = argv[++i];
    }
    else if (arg == "-l" || arg == "--language")
    {
      args.language = argv[++i];
    }
    else if (arg == "-r" || arg == "--reduce_logits")
    {
      args.reduce_logits = true;
    }
    else if (arg == "-ds" || arg == "--do_sample")
    {
      args.do_sample = true;
    }
    else if (arg == "-tk" || arg == "--top_k")
    {
      args.top_k = std::stoi(argv[++i]);
    }
    else if (arg == "-tp" || arg == "--top_p")
    {
      args.top_p = std::stof(argv[++i]);
    }
    else if (arg == "-temp" || arg == "--temperature")
    {
      args.temp = std::stof(argv[++i]);
    }
    else if (arg == "-sd" || arg == "--seed")
    {
      args.seed = std::stoi(argv[++i]);
    }
    else if (arg == "-rp" || arg == "--repeat_penalty")
    {
      args.repeat_penalty = std::stof(argv[++i]);
    }
    else if (arg == "-rln" || arg == "--repeat_last_n")
    {
      args.repeat_last_n = std::stoi(argv[++i]);
    }

    else if (arg == "-f" || arg == "--force_max_generation")
    {
      args.force_max_generation = true;
    }
    else if (arg == "-n" || arg == "--num_iteration")
    {
      args.num_iteration = std::stoi(argv[++i]);
    }
    else if (arg == "-v" || arg == "--verbose")
    {
      args.verbose = true;
    }
    else if (arg == "-ell" || arg == "--enable_latency_log")
    {
      args.enable_latency_log = true;
    }
    else
    {
      std::cerr << "Unknown argument: " << arg << std::endl;
      usage(argv[0]);
      exit(EXIT_FAILURE);
    }
  }

  return args;
}

static auto parse_args(int argc, char **argv) -> Args
{
  std::vector<std::string> argv_vec;
  argv_vec.reserve(argc);

#ifdef _WIN32
  LPWSTR *wargs = CommandLineToArgvW(GetCommandLineW(), &argc);

  std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>> converter;
  for (int i = 0; i < argc; i++)
  {
    argv_vec.emplace_back(converter.to_bytes(wargs[i]));
  }

  LocalFree(wargs);

#else
  for (int i = 0; i < argc; i++)
  {
    argv_vec.emplace_back(argv[i]);
  }
#endif

  return parse_args(argv_vec);
}

int32_t get_out_token_id(const std::vector<int> &input_ids, float *logits, size_t vocab_size, Args args)
{
  int32_t out_token;

  // logits pre-process
  if (args.repeat_penalty != 1.f)
  {
    const int penalty_tokens_used_size = std::min((int)input_ids.size(), args.repeat_last_n);
    if (penalty_tokens_used_size)
    {
      sampling_repetition_penalty(logits, logits + vocab_size, input_ids, penalty_tokens_used_size, args.repeat_penalty);
    }
  }

  if (args.do_sample)
  {
    if (args.temp > 0)
    {
      sampling_temperature(logits, logits + vocab_size, args.temp);
    }

    std::vector<TokenIdScore> token_scores(vocab_size);
    for (size_t i = 0; i < vocab_size; i++)
    {
      token_scores[i] = TokenIdScore((int)i, logits[i]);
    }

    // top_k sampling
    if (0 < args.top_k && args.top_k < (int)token_scores.size())
    {
      sampling_top_k(token_scores.data(), token_scores.data() + args.top_k,
                     token_scores.data() + token_scores.size());
      token_scores.resize(args.top_k);
    }

    // top_p sampling
    if (0.f < args.top_p && args.top_p < 1.f)
    {
      auto pos = sampling_top_p(token_scores.data(), token_scores.data() + token_scores.size(), args.top_p);
      token_scores.resize(pos - token_scores.data());
    }

    // sample next token
    sampling_softmax_inplace(token_scores.data(), token_scores.data() + token_scores.size());
    for (size_t i = 0; i < token_scores.size(); i++)
    {
      logits[i] = token_scores[i].score;
    }

    thread_local std::mt19937 rng((size_t)args.seed);

    std::discrete_distribution<> dist(logits, logits + token_scores.size());
    out_token = token_scores[dist(rng)].id;
  }
  else
  {
    out_token = std::max_element(logits, logits + vocab_size) - logits;
  }

  return out_token;
}

int main(int argc, char **argv)
{
  try
  {
    Args args = parse_args(argc, argv);
    qwen::QwenConfig config;
    double total_time;
    std::string file_path = "performance_test_case.txt";
    std::cout << "Read file from disk: " << file_path << "\n";
    std::ifstream inputFile(file_path);
     // Check if the file is successfully opened 
    if (!inputFile.is_open()) { 
	std::cerr << "Error opening the file!" << std::endl; 
        return 1; 
    } 
  
    std::string line; // Declare a string variable to store each 
                      // line of the file 
  
    // Read each line of the file and print it to the 
    // standard output stream 
    std::cout << "File Content: " << std::endl;
    std::stringstream ss; 
    while (getline(inputFile, line)) { 
	    //std::cout << line << std::endl; // Print the current line 
	    ss << line;
    } 
    std::string input_text = ss.str();
    // Close the file 
    inputFile.close();
    std::cout << "Set input prompt as input text: " << input_text << "\n";
    args.prompt = input_text;
    // Init Tokenizer
    auto startTime = Time::now();
    std::unique_ptr<qwen::QwenTokenizer> tokenizer = std::make_unique<qwen::QwenTokenizer>(args.tiktoken_path, config);
    auto duration_ms = get_duration_ms_until_now(startTime);
    std::cout << "Load Qwen tokenizer took " << duration_ms << " ms" << std::endl;

    // Init Text Streamer
    auto text_streamer = std::make_shared<qwen::TextStreamer>(std::cout, tokenizer.get());
    startTime = Time::now();

    // Init OpenVINO Runtime
    ov::Core core;
    std::cout << "Init OpenVINO with version: \n"
              << ov::get_openvino_version() << std::endl;
    ov::AnyMap device_config = {};
    if (args.device.find("CPU") != std::string::npos)
    {
      device_config[ov::cache_dir.name()] = "llm-cache";
      device_config[ov::hint::scheduling_core_type.name()] = ov::hint::SchedulingCoreType::PCORE_ONLY;
      device_config[ov::hint::enable_hyper_threading.name()] = false;
      device_config[ov::hint::enable_cpu_pinning.name()] = true;
      device_config[ov::enable_profiling.name()] = false;
    }

    if (args.device.find("GPU") != std::string::npos)
    {
      device_config[ov::cache_dir.name()] = "llm-cache";
      device_config[ov::intel_gpu::hint::queue_throttle.name()] = ov::intel_gpu::hint::ThrottleLevel::MEDIUM;
      device_config[ov::intel_gpu::hint::queue_priority.name()] = ov::hint::Priority::MEDIUM;
      device_config[ov::intel_gpu::hint::host_task_priority.name()] = ov::hint::Priority::HIGH;
      device_config[ov::hint::enable_cpu_pinning.name()] = true;
      device_config[ov::enable_profiling.name()] = false;
      device_config[ov::intel_gpu::hint::enable_sdpa_optimization.name()] = true;
    }
    constexpr size_t BATCH_SIZE = 1;
    // Model preparation, apply graph optimization on base model
    if (args.reduce_logits)
    {
      // Read OpenVINO Model
      std::shared_ptr<ov::Model> model = core.read_model(args.model_path);
      duration_ms = get_duration_ms_until_now(startTime);
      std::cout << "Read Qwen Model took " << duration_ms << " ms" << std::endl;
      std::vector<ov::Output<ov::Node>> inputs = model->inputs();
      inputs = model->inputs();
      auto outputs = model->outputs();
      // Modify model input type to algin with tokenizer outputs with PrePostProcessor
      std::cout << "######## [Model Graph Optimization] Step 1: Modify model input element type from i64 to i32 ########\n";
      ov::preprocess::PrePostProcessor p3(model);
      p3.input("input_ids").tensor().set_element_type(ov::element::i32); // cast to the type of tokenizer's output
      p3.input("attention_mask").tensor().set_element_type(ov::element::i32);
      p3.input("position_ids").tensor().set_element_type(ov::element::i32);
      // Change input past key value and output present key value with FP16
      model = p3.build();
      std::cout << "######## [Model Graph Optimization] Step 2: Insert slice node after reshape to reduce logits operation ########\n";
      ov::pass::Manager manager;
      manager.register_pass<InsertSlice>();
      manager.run_passes(model);
      std::string modifiled_file = std::regex_replace(args.model_path, std::regex("openvino_model"), "modified_openvino_model");
      std::cout << "######## [Model Graph Optimization] Step 3: Save modified model in " << modifiled_file << " ########\n";
      ov::serialize(model, modifiled_file);
      // Compile modified model from disk to create model cache
      startTime = Time::now();
      ov::CompiledModel compiled_model = core.compile_model(modifiled_file, args.device, device_config);
      duration_ms = get_duration_ms_until_now(startTime);
      std::cout << "Compile model and save model cache took: " << duration_ms << " ms" << std::endl;
      return 0;
    }
    // Direct compile modified Qwen model with FP16 KV input & output + Reduced logits operations [1,1,151936]
    else
    {
      startTime = Time::now();
      ov::CompiledModel compiled_model = core.compile_model(args.model_path, args.device, device_config);
      duration_ms = get_duration_ms_until_now(startTime);
      std::cout << "Compile model took: " << duration_ms << " ms" << std::endl;
      ov::InferRequest ireq = compiled_model.create_infer_request();
      auto model_inputs = compiled_model.inputs();
      int32_t out_token;
      int sentence_num = 0;
      std::vector<std::string> sentences;
      std::vector<std::vector<double>> second_latencies;
      second_latencies.resize(sentences.size());

      if (!args.prompt.empty())
      {
        sentences = {args.prompt};
      }
      else
      {
        if (args.language.find("ch") != std::string::npos)
        {
          sentences = chinese_sentences;
        }
        else if (args.language.find("en") != std::string::npos)
        {
          sentences = english_sentences;
        }
      }
      for (std::string input_text : sentences)
      {

        std::vector<double> second_latency_per_setences;
        // Build input prompt with prompt template
        std::cout << "******************************************* Text Sentence #" << sentence_num << " Start *******************************************\n";
        startTime = Time::now();
        std::vector<int> input_ids = tokenizer->encode_history({input_text}, args.max_length);
        duration_ms = get_duration_ms_until_now(startTime);
        for (int i = 0; i < args.num_iteration; i++)
        {
          text_streamer->put(input_ids);
          std::vector<int> output_ids = input_ids;

          // Prepare input tensor for first infer
          ireq.get_tensor("input_ids").set_shape({BATCH_SIZE, input_ids.size()});
          ireq.get_tensor("attention_mask").set_shape({BATCH_SIZE, input_ids.size()});
          std::copy_n(input_ids.data(), input_ids.size(), ireq.get_tensor("input_ids").data<int32_t>());
          std::fill_n(ireq.get_tensor("attention_mask").data<int32_t>(), input_ids.size(), 1);

          // If stateful model contains 3 inputs (input_ids, attention_mask, beam_idx, position_id), use WA to set beam_idx batch=1, value as 0 for greedy search case
          ireq.get_tensor("beam_idx").set_shape({BATCH_SIZE});
          ireq.get_tensor("beam_idx").data<int32_t>()[0] = 0;
          ireq.get_tensor("position_ids").set_shape({BATCH_SIZE, input_ids.size()});
          std::iota(ireq.get_tensor("position_ids").data<int32_t>(), ireq.get_tensor("position_ids").data<int32_t>() + ireq.get_tensor("position_ids").get_size(), 0);
          for (auto &&state : ireq.query_state())
          {
            state.reset();
          }

          std::cout << "Input token length: " << input_ids.size() << "\n";
          int max_context_length = args.max_context_length;
          if (sentence_num == 4)
          {
            max_context_length = args.max_context_length * 2;
          }
          else
          {
            max_context_length = args.max_context_length;
          }

          // First inference
          startTime = Time::now();
          ireq.infer();
          duration_ms = get_duration_ms_until_now(startTime);
          std::cout << "First inference took " << duration_ms << " ms" << std::endl;

          // Get first inference results
          size_t vocab_size = ireq.get_tensor("logits").get_shape().back();
          float *logits = ireq.get_tensor("logits").data<float>();
          out_token = get_out_token_id(output_ids, logits, vocab_size, args);
          output_ids.emplace_back(out_token);

          if (text_streamer)
          {
            text_streamer->put({out_token});
          }
          ireq.get_tensor("input_ids").set_shape({BATCH_SIZE, 1});
          ireq.get_tensor("position_ids").set_shape({BATCH_SIZE, 1});
          total_time = 0;
          int count = 1;
          bool stop_condition = true;
          while (stop_condition)
          {
            if (args.force_max_generation)
            {
              stop_condition = count < max_context_length - 1;
            }
            else
            {
              stop_condition = out_token != config.eos_token_id && out_token != config.im_end_id && count < max_context_length - 1;
            }
            // Prepare input tensor for 2nd+ inference
            ireq.get_tensor("input_ids").data<int32_t>()[0] = out_token;
            ireq.get_tensor("attention_mask").set_shape({BATCH_SIZE, ireq.get_tensor("attention_mask").get_shape()[1] + 1});
            std::fill_n(ireq.get_tensor("attention_mask").data<int32_t>(), ireq.get_tensor("attention_mask").get_size(), 1);
            ireq.get_tensor("position_ids").data<int32_t>()[0] = int32_t(ireq.get_tensor("attention_mask").get_size() - 2);

            // 2nd+ inference
            startTime = Time::now();
            ireq.infer();
            duration_ms = get_duration_ms_until_now(startTime);
            count += 1;
            // Get 2nd+ inference results
            logits = ireq.get_tensor("logits").data<float>();
            out_token = get_out_token_id(output_ids, logits, vocab_size, args);
            output_ids.emplace_back(out_token);
            if (text_streamer)
            {
              text_streamer->put({out_token});
            }
            total_time += duration_ms;
            if (args.enable_latency_log)
            {
              second_latency_per_setences.push_back(duration_ms);
            };
          }
          if (args.enable_latency_log)
          {
            second_latencies.push_back(second_latency_per_setences);
          };
          std::cout << '\n';
          if (count > 1)
          {
            std::cout << "Other inference took in total: " << total_time << " ms, Average other token latency: " << total_time / (count - 1) << " ms" << std::endl;
            std::cout << "Input num tokens: " << input_ids.size() << ", output num tokens: " << count << ", Average inference speed: " << (count - 1) / total_time * 1000.0 << " token/s\n";
          }
          std::cout << "******************************************* Text Sentence #" << sentence_num << " Finished ****************************************\n\n";
        }
        sentence_num += 1;
      }
      if (text_streamer)
      {
        text_streamer->end();
      }
      if (args.enable_latency_log)
      {
        int i = 0;
        std::ofstream myfile;
        myfile.open("second_latency_each_sentences.txt");
        for (auto latencies : second_latencies)
        {
          myfile << "Second token Latency of sentence number:  " << i << "\n";
          for (auto latency : latencies)
          {
            myfile << latency << "\n";
          }
          i++;
        }
        myfile.close();
      }
    }
  }
  catch (std::exception &e)
  {
    std::cerr << e.what() << std::endl;
    exit(EXIT_FAILURE);
  }

  return 0;
}
