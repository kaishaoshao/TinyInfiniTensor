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
        size_t A_rows = A[dimA - 2]; // 矩阵的行数
        size_t A_cols = A[dimA - 1];
        size_t B_rows = B[dimB - 2];
        size_t B_cols = B[dimB - 1]; // 矩阵的列数

        if(A_cols != B_rows)
            return std::nullopt;   // A 的列数必须等于 B 的行数

        // 处理批量维度
        Shape batchDimsA(A.begin(), A.end() - 2); // 矩阵A的批量维度
        Shape batchDimsB(B.begin(), B.end() - 2);
        // 检查批量维度是否匹配或可广播
        if (batchDimsA != batchDimsB) {
          if (batchDimsA.size() != batchDimsB.size()) {
            return std::nullopt; // Batch dimension count mismatch
          }
          for (size_t i = 0; i < batchDimsA.size(); ++i) {
            if (batchDimsA[i] != batchDimsB[i] && batchDimsA[i] != 1 &&
                batchDimsB[i] != 1) {
              return std::nullopt; // Batch dimensions cannot be broadcasted
            }
          }
        }

        // 计算输出形状
        Shape outputShape;
        for (size_t i = 0; i < batchDimsA.size(); ++i) 
        {
          std::cout << "i: " << i << std::endl;
          std::cout << "batchDimsA[i]: " << batchDimsA[i] << std::endl;
          std::cout << "batchDimsB[i]: " << batchDimsB[i] << std::endl;
          std::cout << "max(batchDimsA[i], batchDimsB[i]): "
                    << std::max(batchDimsA[i], batchDimsB[i]) << std::endl;     
          outputShape.push_back(std::max(batchDimsA[i], batchDimsB[i])); // 广播后的批量维度
        }
        outputShape.push_back(A_rows); // 输出矩阵的行数
        outputShape.push_back(B_cols); // 输出矩阵的列数

        // 打印输出形状
        std::cout << "Output Shape: ";
        for (int dim : outputShape) {
          std::cout << dim << " ";
        }
        std::cout << std::endl;
       
        return {{outputShape}};
    }

} // namespace infini