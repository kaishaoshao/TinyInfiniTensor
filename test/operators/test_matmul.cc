#include "core/graph.h"
#include "core/kernel.h"
#include "core/runtime.h"
#include "operators/matmul.h"

#include "test.h"

namespace infini
{
    using ExpectOutput = vector<float>;

    TEST(Matmul, ShapeInference)
    {
        auto runtime = NativeCpuRuntimeObj::getInstance();
        {
            Graph g = make_ref<GraphObj>(runtime);
            auto A = g->addTensor(Shape{1, 3, 5});
            auto B = g->addTensor(Shape{1, 5, 2});
            auto matmul = g->addOp<MatmulObj>(A, B, nullptr);
            std::cout << "Output tensor matmul : " << matmul << std::endl;
            auto C = matmul->getOutputs()[0];
            std::cout << "Output tensor C: " << C << std::endl; // 打印输出张量
            EXPECT_EQ(C->getDims(), (Shape{1, 3, 2}));
        }
        {
            Graph g = make_ref<GraphObj>(runtime);
            auto A = g->addTensor(Shape{3, 5, 4});
            auto B = g->addTensor(Shape{3, 5, 2});
            auto matmul = g->addOp<MatmulObj>(A, B, nullptr, true, false);
            auto C = matmul->getOutputs()[0];
            std::cout << "Output tensor C: " << C << std::endl; // 打印输出张量
            EXPECT_EQ(C->getDims(), (Shape{3, 4, 2}));
        }
        {
            Graph g = make_ref<GraphObj>(runtime);
            auto A = g->addTensor(Shape{1, 2, 3, 5});
            auto B = g->addTensor(Shape{1, 1, 5, 2});
            auto matmul = g->addOp<MatmulObj>(A, B, nullptr);
            auto C = matmul->getOutputs()[0];
            EXPECT_EQ(C->getDims(), (Shape{1, 2, 3, 2}));
        }
        {
            Graph g = make_ref<GraphObj>(runtime);
            auto A = g->addTensor(Shape{2, 3, 5, 4});
            auto B = g->addTensor(Shape{1, 3, 5, 2});
            auto matmul = g->addOp<MatmulObj>(A, B, nullptr, true, false);
            auto C = matmul->getOutputs()[0];
            EXPECT_EQ(C->getDims(), (Shape{2, 3, 4, 2}));
        }
        {
            Graph g = make_ref<GraphObj>(runtime);
            auto A = g->addTensor(Shape{2, 3, 5, 4});
            auto B = g->addTensor(Shape{1, 3, 2, 5});
            auto matmul = g->addOp<MatmulObj>(A, B, nullptr, true, true);
            auto C = matmul->getOutputs()[0];
            EXPECT_EQ(C->getDims(), (Shape{2, 3, 4, 2}));
        }
    }

}; // namespace infini