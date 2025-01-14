#include "operators/transpose.h"

namespace infini
{
    TransposeObj::TransposeObj(GraphObj *graph, Tensor input, Tensor output,
                               vector<int> permute)
        : OperatorObj(OpType::Transpose, {input}, {output})
    {
        auto rank = input->getRank();
        if (permute.empty())
        {
            for (size_t i = 0; i < rank; ++i)
            {
                transposePermute[i] = i;
            }
        }
        else
        {
            IT_ASSERT(rank == permute.size());
            transposePermute = std::move(permute);
        }
        IT_ASSERT(checkValid(graph));
    }

    /*
    假设输入张量的形状为 [1, 2, 3]：
    如果 perm_ 为空，转置后的形状为 [3, 2, 1]。
    如果 perm_ 为 [1, 0, 2]，转置后的形状为 [2, 1, 3]。
    */

    optional<vector<Shape>> TransposeObj::inferShape(const TensorVec &inputs)
    {
        const auto A = inputs[0];
        auto input_dim = A->getDims(); // 输入张量的 shape
        auto output_dim = input_dim;   // 输出张量的 shape 初始化形状
        int rank = A->getRank();       // 输出张量的秩

        // =================================== 作业 ===================================
        // TODO：修改 output_dim，返回正确的 transpose 后的 shape
        // REF: https://onnx.ai/onnx/operators/onnx__Transpose.html#transpose-21
        // =================================== 作业 ===================================
        std::cout << "Input dim: ";
        for (const auto &dim : input_dim) {
          std::cout << dim << " ";
        }
        std::cout << std::endl;

        // 如果 permute 为空，则默认将输入张量翻转
        auto permute = getPermute();
        std::cout << "Permute: ";
        for (const auto &p : permute) {
          std::cout << p << " ";
        }
        std::cout << std::endl;

        if (permute.empty())
        {
          std::cout << "Permute is empty, reversing dimensions" << std::endl;
          std::reverse(output_dim.begin(), output_dim.end());
        }
        else
        {
          // 检查 permute 的长度是否与输入张量的秩相同
          if (static_cast<int>(permute.size()) != rank)
          {
              std::cout << "Permute length does not match input rank"
                      << std::endl;
              return {{output_dim}};
          }
          std::cout << "test" << std::endl;
          for (int i = 0; i < rank; i++) 
          {
            // 检查 permute 中的每个元素是否在输入张量的范围内
            if (permute[i] < 0 || permute[i] >= rank)
            {
              std::cout << "Invalid permute value: " << permute[i] << std::endl;
              return {{output_dim}};
            }

          }
            // 根据 transposePermute 指定的维度顺序重新排列维度
            for (int i = 0; i < rank; i++)
                output_dim[i] = input_dim[permute[i]];
        }


        return {{output_dim}};
    }

    std::string TransposeObj::toString() const
    {
        std::ostringstream os;
        os << type.toString() << "[" << getGuid() << "]";
        os << "(";
        os << vecToString(inputs[0]->getDims()) << ",";
        os << "input=" << inputs[0]->getGuid() << ",";
        os << "output=" << outputs[0]->getGuid() << ")";
        return os.str();
    }
}; // namespace infini
