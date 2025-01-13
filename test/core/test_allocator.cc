#include "core/graph.h"
#include "core/kernel.h"
#include "core/runtime.h"
#include "operators/unary.h"

#include "test.h"

using namespace std;

namespace infini
{
/*
测试目的
测试内存分配器的基本功能，包括内存分配和释放。

验证内存分配器是否能够正确重用释放的内存。

测试逻辑
分配三个张量 a、b、c 的内存，分别记录它们的偏移量 offsetA、offsetB、offsetC。

释放 b 的内存。

分配 d 的内存，记录偏移量 offsetD。

检查 offsetD 是否等于 offsetB，验证内存分配器是否重用了 b 的内存。

检查所有偏移量是否有效（不为 0）。

用意
验证内存分配器是否能够正确管理内存块，并在释放后重用内存。

确保内存分配器不会返回无效的偏移量（如 0）。
 */
TEST(Allocator, testAlloc) {
  Shape shape = Shape{1, 2, 2, 3};
  Runtime runtime = NativeCpuRuntimeObj::getInstance();
  Tensor a = make_ref<TensorObj>(shape, DataType::Float32, runtime);
  Tensor b = make_ref<TensorObj>(shape, DataType::Float32, runtime);
  Tensor c = make_ref<TensorObj>(shape, DataType::Float32, runtime);
  Tensor d = make_ref<TensorObj>(shape, DataType::Float32, runtime);
  Allocator allocator = Allocator(runtime);
  // allocate a->b->c
  size_t offsetA = allocator.alloc(a->getBytes());
  size_t offsetB = allocator.alloc(b->getBytes());
  size_t offsetC = allocator.alloc(c->getBytes());

  // free b, then allocate d
  allocator.free(offsetB, b->getBytes());

  size_t offsetD = allocator.alloc(d->getBytes());

  // expected to be a->d->c
  EXPECT_EQ(offsetB, offsetD);

  ASSERT_FALSE(offsetA == 0 && offsetB == 0 && offsetC == 0 && offsetD == 0);
    }
    /*
    测试目的
    测试内存分配器在释放末尾内存块后的行为。

    验证内存分配器是否能够正确重用释放的末尾内存块。

    测试逻辑
    分配三个张量 a、b、c 的内存，记录 c 的偏移量 offsetC。

    释放 c 的内存。

    分配 d 的内存，记录偏移量 offsetD。

    检查 offsetD 是否等于 offsetC，验证内存分配器是否重用了 c 的内存。

    用意
    验证内存分配器是否能够正确处理末尾内存块的释放和重用。

    确保内存分配器在释放末尾内存块后不会产生内存碎片。


    */
    TEST(Allocator, testAllocWithEndFreeBlock)
    {
        Shape shape = Shape{1, 2, 2, 3};
        Runtime runtime = NativeCpuRuntimeObj::getInstance();
        Tensor a = make_ref<TensorObj>(shape, DataType::Float32, runtime);
        Tensor b = make_ref<TensorObj>(shape, DataType::Float32, runtime);
        Tensor c = make_ref<TensorObj>(shape, DataType::Float32, runtime);
        Tensor d =
            make_ref<TensorObj>(Shape{2, 2, 2, 3}, DataType::Float32, runtime);
        Allocator allocator = Allocator(runtime);
        // allocate a->b->c
        allocator.alloc(a->getBytes());
        allocator.alloc(b->getBytes());
        size_t offsetC = allocator.alloc(c->getBytes());
        allocator.info();
        // free c, then allocate d
        allocator.free(offsetC, c->getBytes());
        allocator.info();
        size_t offsetD = allocator.alloc(d->getBytes());
        allocator.info();
        // expected to be a->b->d, with no free block between b and c
        EXPECT_EQ(offsetC, offsetD);
    }
    /*
    测试目的
    测试内存分配器的 getPtr 函数是否能够正确返回分配的内存指针。

    验证多次调用 getPtr 是否返回相同的指针。

    测试逻辑
    分配四个张量 a、b、c、d 的内存。

    多次调用 getPtr 函数，获取分配的内存指针。

    检查多次调用 getPtr 返回的指针是否相同。

    用意
    验证 getPtr 函数的行为是否符合预期，确保其返回的内存指针是稳定的。

    确保内存分配器的内部状态管理正确，不会因为多次调用 getPtr 而返回不同的指针。

    */
    TEST(Allocator, testGetPtr)
    {
        Shape shape = Shape{1, 2, 2, 3};
        Runtime runtime = NativeCpuRuntimeObj::getInstance();
        Tensor a = make_ref<TensorObj>(shape, DataType::Float32, runtime);
        Tensor b = make_ref<TensorObj>(shape, DataType::Float32, runtime);
        Tensor c = make_ref<TensorObj>(shape, DataType::Float32, runtime);
        Tensor d = make_ref<TensorObj>(shape, DataType::Float32, runtime);
        Allocator allocator = Allocator(runtime);
        // allocate a->b->c->d
        allocator.alloc(a->getBytes());
        allocator.alloc(b->getBytes());
        allocator.alloc(c->getBytes());
        allocator.alloc(d->getBytes());
        // multiple calls to the getPtr() function should return the same pointer
        void *ptr1 = allocator.getPtr();
        void *ptr2 = allocator.getPtr();
        EXPECT_EQ(ptr1, ptr2);
    }

} // namespace infini
