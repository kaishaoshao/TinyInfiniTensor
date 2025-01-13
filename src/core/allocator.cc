#include "core/allocator.h"
#include <utility>

namespace infini
{
    Allocator::Allocator(Runtime runtime) : runtime(runtime)
    {
        used = 0;
        peak = 0;
        ptr = nullptr;

        // 'alignment' defaults to sizeof(uint64_t), because it is the length of
        // the longest data type currently supported by the DataType field of
        // the tensor
        alignment = sizeof(uint64_t);
    }

    Allocator::~Allocator()
    {
        if (this->ptr != nullptr)
        {
            runtime->dealloc(this->ptr);
        }
    }

    size_t Allocator::alloc(size_t size)
    {
        IT_ASSERT(this->ptr == nullptr);
        // pad the size to the multiple of alignment
        size = this->getAlignedSize(size);
        std::cout << "Allocator::alloc size: " << size << std::endl;
        // =================================== 作业 ===================================
        // TODO: 设计一个算法来分配内存，返回起始地址偏移量
        // =================================== 作业 ===================================
        for (auto it = free_blocks.begin(); it != free_blocks.end(); ++it)
        {
            size_t start_addr = it->first;
            size_t block_size = it->second + start_addr;
            std::cout << "size: " << size << ", block_size111: " << block_size << std::endl;
            if (block_size >= size)
            {
                // 分配内存
                free_blocks.erase(it); // 移除当前空闲块
            
                if (block_size > size)
                    free_blocks[start_addr+size] = block_size - size;

                // 更新已用内存和峰值内存
                used += size;
                if (used > peak)
                {
                    peak = used;
                }
                std::cout << "Allocator::alloc1: " << start_addr << " " << size
                          << std::endl;
                return start_addr; 
            }
            
        }

        // 如果没有足够的内存，返回 0（表示分配失败）
        size_t start_addr = peak;
        peak += size;
        used += size;
        std::cout << "Allocator::alloc2: " << start_addr << " " << size << std::endl;
        return start_addr;
    }

    void Allocator::free(size_t addr, size_t size)
    {
        IT_ASSERT(this->ptr == nullptr);
        size = getAlignedSize(size);
        // =================================== 作业 ===================================
        // TODO: 设计一个算法来回收内存
        // =================================== 作业 ===================================

        // 插入释放的内存块
        free_blocks[addr] = size;

        // 合并相邻的空闲块
        // 检查前一个块
        auto prev_it = free_blocks.find(addr);
        if (prev_it != free_blocks.begin())
        {
            prev_it--; // 移动到前一个块
            if (prev_it->first + prev_it->second == addr)
            {
                size_t prev_size = prev_it->second;
                free_blocks.erase(prev_it);
                addr -= prev_size;
                size += prev_size;
                std::cout << "free1: " << prev_size << std::endl;
            }
        }

        // 检查后一个块
        auto next_it = free_blocks.find(addr + size);
        if (next_it != free_blocks.end())
        {
            // 合并后一个块
            size += next_it->second;
            free_blocks.erase(next_it);
            std::cout << "free2: " << size << std::endl;
        }

        // 插入合并后的空闲块
        std::cout << "size1: " << addr << std::endl;
        free_blocks[addr] = size;
        std::cout << "free3: " << size << std::endl;
        // 更新已用内存
        used -= size;
    }

    void *Allocator::getPtr()
    {
        if (this->ptr == nullptr)
        {
            this->ptr = runtime->alloc(this->peak);
            printf("Allocator really alloc: %p %lu bytes\n", this->ptr, peak);
        }
        return this->ptr;
    }

    size_t Allocator::getAlignedSize(size_t size)
    {
        return ((size - 1) / this->alignment + 1) * this->alignment;
    }

    void Allocator::info()
    {
        std::cout << "Used memory: " << this->used << ", peak memory: " << this->peak
                  << std::endl;
    }
} // namespace infini
