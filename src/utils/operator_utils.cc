#include "utils/operator_utils.h"
#include "core/runtime.h"

namespace infini {

Shape infer_broadcast(const Shape &A, const Shape &B) {

    // =================================== 作业 ===================================
    // TODO：对 A 和 B 进行双向广播，返回广播后的形状。
    // REF: https://github.com/onnx/onnx/blob/main/docs/Broadcasting.md
    // =================================== 作业 ===================================
    if (A.size() == 0 && B.size() == 0)
        return {};
    std::cout << "Input A: ";
    for (int dim : A)
      std::cout << dim << " ";
    std::cout << std::endl;

    std::cout << "Input B: ";
    for (int dim : B)
      std::cout << dim << " ";
    std::cout << std::endl;

    auto rankA = A.size();
    auto rankB = B.size();
    auto maxRank = std::max(rankA, rankB);

    // 对齐形状的维度(在较少的形状前面补1)
    Shape paddedA(maxRank, 1);
    Shape paddedB(maxRank, 1);

    std::copy(A.rbegin(), A.rend(), paddedA.rbegin());
    std::copy(B.rbegin(), B.rend(), paddedB.rbegin());

    // 打印对齐后的形状
    std::cout << "Padded A: ";
    for (int dim : paddedA)
      std::cout << dim << " ";
    std::cout << std::endl;

    std::cout << "Padded B: ";
    for (int dim : paddedB)
      std::cout << dim << " ";
    std::cout << std::endl;

    // 计算广播后形状
    Shape result(maxRank);
    for (size_t i = 0; i < maxRank; i++)
    {
        int dimA = paddedA[i];
        int dimB = paddedB[i];

        // 检查是否可以广播
        if(dimA != dimB && dimA != 1 && dimB != 1)
            throw std::invalid_argument("Cannot broadcast shapes " +
                                        std::to_string(dimA) + " and " +
                                        std::to_string(dimB));
        result[i] = std::max(dimA, dimB);
    }

    std::cout << "Broadcasted Shape: ";
    for (int dim : result)
      std::cout << dim << " ";
    std::cout << std::endl;

    return {result};
}

int get_real_axis(const int &axis, const int &rank) {
    IT_ASSERT(rank >= 1);
    IT_ASSERT(axis >= -rank && axis <= (rank - 1));
    int newAxis;
    if (axis < 0) {
        newAxis = rank + axis;
    } else {
        newAxis = axis;
    }
    return newAxis;
}

Shape locate_index(size_t inputN, const Shape &shape) {
    Shape ans(shape.size());
    auto i = ans.rbegin();
    auto j = shape.rbegin(), ej = shape.rend();
    while (j != ej) {
        auto div = std::div(inputN, *j++);
        *i++ = div.rem;
        inputN = div.quot;
    }
    return ans;
}

size_t delocate_index(const Shape &shapeIndex, const Shape &shape,
                      const Shape &stride) {
    size_t ans = 0;
    Shape index(shapeIndex.size());
    IT_ASSERT(shapeIndex.size() == shape.size());
    IT_ASSERT(shape.size() == stride.size());
    for (size_t i = 0; i < shape.size(); ++i) {
        index[i] = shapeIndex[i] % shape[i];
        ans += index[i] * stride[i];
    }
    return ans;
}

std::string device_to_str(Device device) {
    std::string deviceStr;
    switch (device) {
    case Device::CPU:
        return "CPU";
    default:
        IT_TODO_HALT();
    }
}

std::string get_kernel_attrs_str(const KernelAttrs &kernelAttrs) {
    std::string deviceStr = device_to_str(std::get<0>(kernelAttrs));
    std::string opStr = OpType(std::get<1>(kernelAttrs)).toString();
    return deviceStr + ", " + opStr;
}

} // namespace infini
