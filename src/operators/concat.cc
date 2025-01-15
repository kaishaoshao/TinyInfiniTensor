#include "operators/concat.h"
#include "utils/operator_utils.h"

namespace infini {
ConcatObj::ConcatObj(GraphObj *graph, TensorVec inputs, Tensor output, int _dim)
    : OperatorObj(OpType::Concat, inputs, {output}) {
    int rank = inputs[0]->getRank();
    dim = get_real_axis(_dim, rank);
    IT_ASSERT(checkValid(graph));
}

/*
Concat
操作符用于将一组输入张量沿指定轴连接成一个张量。所有输入张量的形状必须相同，除了连接轴的维度大小可以不同。

关键属性
axis（INT，必需）：
指定沿哪个轴连接张量。
负值表示从后向前计数维度。
可接受范围：[-r, r-1]，其中 r 是输入张量的秩。

输入
inputs（可变，异构）：
1 到 2147483647 个输入张量。
所有输入张量的形状必须相同，除了连接轴的维度大小可以不同。

输出
concat_result（异构）：
连接后的张量。
 */
optional<vector<Shape>> ConcatObj::inferShape(const TensorVec &inputs) {
    Shape dims = inputs[0]->getDims();
    auto rank = inputs[0]->getRank();

    // =================================== 作业 ===================================
    // TODO：修改 dims，返回正确的 concat 后的 shape
    // REF: https://onnx.ai/onnx/operators/onnx__Concat.html#concat-13
    // =================================== 作业 ===================================
    if (inputs.empty())
        return {};
    
    int axis = get_real_axis(this->dim, rank);
    if (axis < 0 || axis >= (int)rank)   
    {
        return {};
    }

    for (size_t i = 1; i < inputs.size(); ++i)
    {
        // 检查输入张量rank是否相同
        if (inputs[0]->getRank() != rank)
            return {};

        dims[axis] += inputs[i]->getDims()[axis];
        
    }
    return {{dims}};
       
}

std::string ConcatObj::toString() const {
    std::ostringstream os;
    os << "Concat[" << getGuid() << "]";
    os << "(";
    for (auto input : inputs)
        os << vecToString(input->getDims()) << ",";
    os << "dim=" << dim << ",";
    os << "input=";
    for (auto input : inputs)
        os << input->getGuid() << ",";
    os << "output=" << outputs[0]->getGuid() << ")";
    return os.str();
}

} // namespace infini
