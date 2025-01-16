#include "operators/matmul.h"

namespace infini
{

    MatmulObj::MatmulObj(GraphObj *graph, Tensor A, Tensor B, Tensor C, bool transA,
                         bool transB)
        : OperatorObj(OpType::MatMul, TensorVec{A, B}, {C}),
          transA(transA), transB(transB)
    {
        IT_ASSERT(checkValid(graph));
    }

    string MatmulObj::toString() const
    {
        std::ostringstream os;
        os << "Matmul([" << (transA ? "A^T" : "A") << "," << (transB ? "B^T" : "B]")
           << ",A=" << inputs[0]->getGuid()
           << ",B=" << inputs[1]->getGuid() << ",C=" << outputs[0]->getGuid()
           << ",mnk=[" << m << "," << n << "," << k << "])";
        return os.str();
    }

    optional<vector<Shape>> MatmulObj::inferShape(const TensorVec &inputs)
    {
        // =================================== 作业 ===================================
        // TODO：返回经过 matmul 操作后的 shape
        // REF: https://github.com/onnx/onnx/blob/main/docs/Operators.md#gemm
        // =================================== 作业 ===================================
        if (inputs.size() != 2)
            return std::nullopt;

        // 获取张量的形状
        Shape A = inputs[0]->getDims();
        Shape B = inputs[1]->getDims();

        // 检查输入张量的形状是否合法
        if (A.size() < 2 || B.size() < 2)
            return std::nullopt;

        // 获取矩阵的维度
        size_t dimA = A.size();
        size_t dimB = B.size();
        size_t A_L = A[dimA - 2]; // 矩阵的行数
        size_t B_L = B[dimB - 2];
        size_t A_W = A[dimA - 1];
        size_t B_W = B[dimB - 1]; // 矩阵的列数

        if(A_W != B_L)
            return std::nullopt;   // A 的列数必须等于 B 的行数

        // 处理批量维度
        Shape batchDimsA(A.begin(), A.end() - 2); // 矩阵A的批量维度
        Shape batchDimsB(B.begin(), B.end() - 2);
        // 检查批量维度是否匹配或可广播
        if (batchDimsA != batchDimsB) {
          // 如果批量维度不匹配，尝试广播
          if (batchDimsA.size() != batchDimsB.size()) {
            return std::nullopt; // 批量维度数量不匹配
          }
          for (size_t i = 0; i < batchDimsA.size(); ++i) {
            if (batchDimsA[i] != batchDimsB[i] && batchDimsA[i] != 1 &&
                batchDimsB[i] != 1) {
              return std::nullopt; // 批量维度无法广播
            }
          }
        }

        // 计算输出形状
        Shape outputShape;
        for (size_t i = 0; i < batchDimsA.size(); ++i) {
          outputShape.push_back(
              std::max(batchDimsA[i], batchDimsB[i])); // 广播后的批量维度
        }
         outputShape.push_back(A_L);
         outputShape.push_back(B_W);
        std::cout<<"outputShape1: " << outputShape[0] <<std::endl;
        std::cout << "outputShape: " << outputShape[1] << std::endl;
        return {outputShape};
    }

} // namespace infini