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

        auto input_A = inputs[0];
        auto input_B = inputs[1];
        
        auto dims_A = input_A->getDims();
        auto dims_B = input_B->getDims();
        auto dims = std::max(dims_A, dims_B);

        auto rank_A = input_A->getRank();
        auto rank_B = input_B->getRank();
        auto rank = std::max(rank_A, rank_B);

        for (size_t i = 0; i < rank - 2; i++)
            dims[i] = std::max(dims_A[i], dims_B[i]);

        if (transA) {
          dims[rank_A - 2] = dims_A[rank_A - 1];
        } else {
          dims[rank_B - 2] = dims_A[rank_B - 2];
        }

        if (transB) {
         dims[rank_A - 1] = dims_B[rank_A - 2];
        } else {
          dims[rank_A - 1] = dims_B[rank_A - 1];
        }

        return {{dims}};
    }

} // namespace infini