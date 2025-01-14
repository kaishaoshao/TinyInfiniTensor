#include "operators/unary.h"

namespace infini
{
    UnaryObj::UnaryObj(OpType type, GraphObj *graph, Tensor input, Tensor output)
        : OperatorObj(type, {input}, {output})
    {
        IT_ASSERT(checkValid(graph));
    }

    optional<vector<Shape>> UnaryObj::inferShape(const TensorVec &inputs)
    {
        const auto A = inputs[0];
        return {{A->getDims()}};
    }

    std::string UnaryObj::toString() const
    {
        std::ostringstream os;
        os << type.toString() << "[" << getGuid() << "]";
        os << "(";
        os << vecToString(inputs[0]->getDims()) << ",";
        os << "input=" << inputs[0]->getGuid() << ",";
        os << "output=" << outputs[0]->getGuid() << ")";
        return os.str();
    }

    ClipObj::ClipObj(GraphObj *graph, Tensor input, Tensor output,
                     std::optional<float> min, std::optional<float> max)
        : OperatorObj(OpType::Clip, {input}, {output}), minValue(min),
          maxValue(max)
    {
        IT_ASSERT(checkValid(graph));
    }

    /*
    Clip 算子分析
    Clip 算子用于将输入张量的元素限制在一个区间内。区间的上下界由 min 和 max
    参数指定。如果元素小于 min，则将其替换为 min；如果元素大于 max，则将其替换为max。

    1. 算子功能
    输入：
    input：输入张量。

    min（可选）：最小值，必须是标量（形状为空的张量）。
    max（可选）：最大值，必须是标量（形状为空的张量）。

    输出：
    output：裁剪后的输出张量。

    默认值：
    如果未提供 min，则默认值为 numeric_limits::lowest()。
    如果未提供 max，则默认值为 numeric_limits::max()。

    特殊情况：
    如果 min > max，则将所有输入值替换为 max。

     */
    optional<vector<Shape>> ClipObj::inferShape(const TensorVec &inputs)
    {
        // =================================== 作业 ===================================
        // TODO：返回经过 clip 操作后的 shape
        // REF: https://onnx.ai/onnx/operators/onnx__Clip.html#clip-13
        // =================================== 作业 ===================================
        // 判断输入张量是否为空
        if (inputs.empty() || !inputs[0])
            return std::nullopt;

        const auto input = inputs[0];
        auto inputShape = input->getDims(); //输入张量的形状

        return {{inputShape}};
    }

    std::string ClipObj::toString() const
    {
        std::ostringstream os;
        os << type.toString() << "[" << getGuid() << "]";
        os << "(";
        os << vecToString(inputs[0]->getDims()) << ",";
        os << "input=" << inputs[0]->getGuid() << ",";
        os << "output=" << outputs[0]->getGuid() << ")";
        return os.str();
    }

    CastObj::CastObj(GraphObj *graph, Tensor input, Tensor output, CastType type)
        : OperatorObj(OpType::Cast, {input}, {output}), castType(type)
    {
        IT_ASSERT(checkValid(graph));
    }

    /*
      Cast 算子分析
      Cast
      算子用于将输入张量的元素转换为指定的数据类型，并返回一个形状相同但数据类型不同的输出张量。
      以下是Cast 算子的详细分析：
        1. 算子功能
        输入：input：输入张量。
        to：目标数据类型（必须为 DataType 枚举中的类型）。

        输出：
        output：转换后的输出张量，形状与输入张量相同。
        数据类型转换规则：
        支持数值类型之间的转换（如 float 转 int）。
        支持字符串与数值类型之间的转换（如 string 转 float）。
        转换时可能会丢失精度或导致值的变化（如 float 转 int 时截断小数部分）。
    */
    vector<DataType> CastObj::inferDataType(const TensorVec &inputs) const
    {
        // =================================== 作业 ===================================
        // TODO：返回经过 cast 操作后, 输出 tensor 的数目和数据类型
        // REF_FILE: src/core/operator.cc
        // REF: https://onnx.ai/onnx/operators/onnx__Cast.html#cast-21
        // =================================== 作业 ===================================
        if(inputs.empty() || !inputs[0])
            return {};

        DataType inputDataType = getOutputDataType();
        return {inputDataType};
    }

    optional<vector<Shape>> CastObj::inferShape(const TensorVec &inputs)
    {
        // =================================== 作业 ===================================
        // TODO：返回经过 cast 操作后的 shape
        // REF: https://onnx.ai/onnx/operators/onnx__Cast.html#cast-21
        // =================================== 作业 ===================================
        if (inputs.empty() || !inputs[0])
            return std::nullopt;

        const auto input = inputs[0];
        auto inputShape = input->getDims();

        return {{inputShape}};
    }

    std::string CastObj::toString() const
    {
        std::ostringstream os;
        os << type.toString() << "[" << getGuid() << "]";
        os << "(";
        os << "output=" << outputs[0]->getGuid() << ")";
        return os.str();
    }

    DataType CastObj::getOutputDataType() const
    {
        switch (castType)
        {
        case CastType::Float2Float16:
            return DataType::Float16;
        case CastType::Float2Int64:
            return DataType::Int64;
        case CastType::Float2Int32:
            return DataType::Int32;
        case CastType::Float2Int16:
            return DataType::Int16;
        case CastType::Float2Int8:
            return DataType::Int8;
        case CastType::Int322Float:
            return DataType::Float32;
        case CastType::Int322Int8:
            return DataType::Int8;
        case CastType::Int322Int16:
            return DataType::Int16;
        case CastType::Int162Float:
            return DataType::Float32;
        case CastType::Int162Int32:
            return DataType::Int32;
        case CastType::Int82Float:
            return DataType::Float32;
        case CastType::Int82Int16:
            return DataType::Int16;
        case CastType::Int82Int32:
            return DataType::Int32;
        case CastType::Uint82Float:
            return DataType::Float32;
        case CastType::Uint82Int32:
            return DataType::Int32;
        case CastType::Uint82Int64:
            return DataType::Int64;
        case CastType::Int322Int64:
            return DataType::Int64;
        case CastType::Int642Int32:
            return DataType::Int32;
        case CastType::Int642Uint32:
            return DataType::UInt32;
        case CastType::Int642Float:
            return DataType::Float32;
        case CastType::Uint322Int64:
            return DataType::Int64;
        case CastType::Float162Float:
            return DataType::Float32;
        case CastType::BFloat162Float:
            return DataType::Float32;
        case CastType::Float2BFloat16:
            return DataType::BFloat16;
        case CastType::Float2Float:
            return DataType::Float32;
        default:
            IT_TODO_HALT();
        }
    }
}; // namespace infini
