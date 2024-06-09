# 1. 关于onnx
commit: b90e252da11dea9bdc191d6b9b8d01511ef3e3bd
tag: v1.9.0
由开发文档知:

Folder structure
- `onnx/`:
    - `onnx.proto`: 定义ONNX所有结构protobuf
    - `checker.py`: 检查序列化的ONNX proto是否合法
    - `shape_inference.py`: ONNX模型的形状和类型的推导
    - `version_converter.py`: 更新或者下载ONNX模型的工具
    - `parser.py`: 从文本表示创建ONNX模型或计算图的工具
    - `hub.py`: 从[ONNX Model Zoo](https://github.com/onnx/models)下载模型的工具
    - `compose.py`: 合并ONNX模型的工具
    - `helper.py`: 计算图操作的工具
    - `defs/`: 定义ONNX算子的文件夹
    - `test/`: 测试文件

## 先查看`onnx.proto3`文件主要定义的类型大概了解ONNX:
```
// 开头是下下面三句话，所以说ONNX是由以下三块构成的
// ONNX is an open specification that is comprised of the following components:
//  
// 1)  A definition of an extensible computation graph model.
// 2)  Definitions of standard data types.
// 3)  Definitions of built-in operators.
```

```
// 定义版本，用作版本控制
enum Version {
  _START_VERSION = 0;
  // which was the version we published on Oct 10, 2017.
  IR_VERSION_2017_10_10 = 0x0000000000000001;

  // IR_VERSION 2 published on Oct 30, 2017
  IR_VERSION_2017_10_30 = 0x0000000000000002;

  // IR VERSION 3 published on Nov 3, 2017
  IR_VERSION_2017_11_3 = 0x0000000000000003;

  // IR VERSION 4 published on Jan 22, 2019
  IR_VERSION_2019_1_22 = 0x0000000000000004;

  // IR VERSION 5 published on March 18, 2019
  IR_VERSION_2019_3_18 = 0x0000000000000005;

  // IR VERSION 6 published on Sep 19, 2019
  IR_VERSION_2019_9_19 = 0x0000000000000006;

  // IR VERSION 7 published on May 8, 2020
  IR_VERSION_2020_5_8 = 0x0000000000000007;

  // IR VERSION 8 published on July 30, 2021
  IR_VERSION_2021_7_30 = 0x0000000000000008;

  // IR VERSION 9 published on May 5, 2023
  IR_VERSION_2023_5_5 = 0x0000000000000009;

  // IR VERSION 10 published on TBD
  IR_VERSION = 0x000000000000000A;
}
```


```
/ 定义属性格式
message AttributeProto {
  reserved 12, 16 to 19;
  reserved "v";

  // Note: this enum is structurally identical to the OpSchema::AttrType
  // enum defined in schema.h.  If you rev one, you likely need to rev the other.
  enum AttributeType {
    UNDEFINED = 0;
    FLOAT = 1;
    INT = 2;
    STRING = 3;
    TENSOR = 4;
    GRAPH = 5;
    SPARSE_TENSOR = 11;
    TYPE_PROTO = 13;

    FLOATS = 6;
    INTS = 7;
    STRINGS = 8;
    TENSORS = 9;
    GRAPHS = 10;
    SPARSE_TENSORS = 12;
    TYPE_PROTOS = 14;
  }

  // The name field MUST be present for this version of the IR.
  string name = 1;           // namespace Attribute

  // if ref_attr_name is not empty, ref_attr_name is the attribute name in parent function.
  // In this case, this AttributeProto does not contain data, and it's a reference of attribute
  // in parent scope.
  // NOTE: This should ONLY be used in function (sub-graph). It's invalid to be used in main graph.
  string ref_attr_name = 21;

  // A human-readable documentation for this attribute. Markdown is allowed.
  string doc_string = 13;

  // The type field MUST be present for this version of the IR.
  // For 0.0.1 versions of the IR, this field was not defined, and
  // implementations needed to use has_field heuristics to determine
  // which value field was in use.  For IR_VERSION 0.0.2 or later, this
  // field MUST be set and match the f|i|s|t|... field in use.  This
  // change was made to accommodate proto3 implementations.
  AttributeType type = 20;   // discriminator that indicates which field below is in use

  // Exactly ONE of the following fields must be present for this version of the IR
  float f = 2;               // float
  int64 i = 3;               // int
  bytes s = 4;               // UTF-8 string
  TensorProto t = 5;         // tensor value
  GraphProto g = 6;          // graph
  SparseTensorProto sparse_tensor = 22;  // sparse tensor value
  // Do not use field below, it's deprecated.
  // optional ValueProto v = 12;         // value - subsumes everything but graph
  TypeProto tp = 14;          // type proto

  repeated float floats = 7;          // list of floats
  repeated int64 ints = 8;            // list of ints
  repeated bytes strings = 9;         // list of UTF-8 strings
  repeated TensorProto tensors = 10;  // list of tensors
  repeated GraphProto graphs = 11;    // list of graph
  repeated SparseTensorProto sparse_tensors = 23; // list of sparse tensors
  repeated TypeProto type_protos = 15;// list of type protos
}
```


```
// 定义ONNX的数据的信息，包括名称，类型，形状
message ValueInfoProto {
  // This field MUST be present in this version of the IR.
  string name = 1;     // namespace Value
  // This field MUST be present in this version of the IR for
  // inputs and outputs of the top-level graph.
  TypeProto type = 2;
  // A human-readable documentation for this value. Markdown is allowed.
  string doc_string = 3;
  // Named metadata values; keys should be distinct.
  repeated StringStringEntryProto metadata_props = 4;
}
```
```
// Node的定义
message NodeProto {
  repeated string input = 1; 
  repeated string output = 2;

  string name = 3;     // namespace Node
  string op_type = 4;  // namespace Operator
  string domain = 7;   // namespace Domain

  string overload = 8;

  repeated AttributeProto attribute = 5;

  string doc_string = 6;

  repeated StringStringEntryProto metadata_props = 9;
}
```

```
// TrainingInfoProto stores information for training a model.
// In particular, this defines two functionalities: an initialization-step
// and a training-algorithm-step. Initialization resets the model
// back to its original state as if no training has been performed.
// Training algorithm improves the model based on input data.
// 存储训练信息, 即包括初始化模型参数，定义一个训练算法计算图用于前向和反向传播，绑定更新操作中的张量名称，计算训练损失，评估模型性能。
message TrainingInfoProto {
  GraphProto initialization = 1; 
  GraphProto algorithm = 2;   // 
  repeated StringStringEntryProto initialization_binding = 3;
  repeated StringStringEntryProto update_binding = 4;
}
```


```
message ModelProto {
  // The version of the IR this model targets.
  int64 ir_version = 1;

  repeated OperatorSetIdProto opset_import = 8; // opset版本

  string producer_name = 2;     // 指示模型是由哪个工具或者框架创建的，比如"pytorch", "tensorflow"
  string producer_version = 3;  // 和上一个字段对应，指示框架或者工具版本

  string domain = 4;    // 表示模型所属的域，用于区分不同的模型类别或应用领域。Together with `model_version` and GraphProto.name, this forms the unique identity of the graph.

  int64 model_version = 5;  // 模型版本

  string doc_string = 6;    // 模型的描述性字符串

  GraphProto graph = 7;     // 模型的计算图结构

  repeated StringStringEntryProto metadata_props = 14;  // 模型的元数据键值对。例如，作者、训练数据集、模型精度等信息

  repeated TrainingInfoProto training_info = 20;    // 模型训练信息，类型TrainingInfoProto上面已经定义过
  repeated FunctionProto functions = 25;          // A list of function protos local to the model.
};
```

```
// 定义基础的key,value对，为了兼容proto2，没有使用map关键字
message StringStringEntryProto {
  string key = 1;
  string value = 2;
};

// tensor注释
message TensorAnnotation {
  string tensor_name = 1;
  // <key, value> pairs to annotate tensor specified by <tensor_name> above.
  // The keys used in the mapping below must be pre-defined in ONNX spec.
  // For example, for 8-bit linear quantization case, 'SCALE_TENSOR', 'ZERO_POINT_TENSOR' will be pre-defined as
  // quantization parameter keys.
  repeated StringStringEntryProto quant_parameter_tensor_names = 2;
}
```

```
// This is the equivalent of the "network" or "graph" in many deep learning frameworks
// 基于一系列node的input和output构成这个有向无环图
message GraphProto {
  // The nodes in the graph, sorted topologically.
  repeated NodeProto node = 1;

  // 计算图名称.
  string name = 2;   // namespace Graph

  // A list of named tensor values, used to specify constant inputs of the graph.
  repeated TensorProto initializer = 5;

  // 以稀疏矩阵存储Initializers.
  repeated SparseTensorProto sparse_initializer = 15;

  string doc_string = 10;   // 模型的描述性字符串

  // 模型输入输出
  repeated ValueInfoProto input = 11;
  repeated ValueInfoProto output = 12;

  // 模型值信息
  repeated ValueInfoProto value_info = 13;

  // This field carries information to indicate the mapping among a tensor and its
  // quantization parameter tensors. For example:
  // For tensor 'a', it may have {'SCALE_TENSOR', 'a_scale'} and {'ZERO_POINT_TENSOR', 'a_zero_point'} annotated,
  // which means, tensor 'a_scale' and tensor 'a_zero_point' are scale and zero point of tensor 'a' in the model.
  repeated TensorAnnotation quantization_annotation = 14;

  // Named metadata values; keys should be distinct.
  repeated StringStringEntryProto metadata_props = 16;

  reserved 3, 4, 6 to 9;
  reserved "ir_version", "producer_version", "producer_tag", "domain";
}
```

```
// A serialized tensor value.
message TensorProto {
  enum DataType {
    UNDEFINED = 0;
    // Basic types.
    FLOAT = 1;   // float
    UINT8 = 2;   // uint8_t
    INT8 = 3;    // int8_t
    UINT16 = 4;  // uint16_t
    INT16 = 5;   // int16_t
    INT32 = 6;   // int32_t
    INT64 = 7;   // int64_t
    STRING = 8;  // string
    BOOL = 9;    // bool

    FLOAT16 = 10;

    DOUBLE = 11;
    UINT32 = 12;
    UINT64 = 13;
    COMPLEX64 = 14;
    COMPLEX128 = 15; 

    BFLOAT16 = 16;

    FLOAT8E4M3FN = 17;
    FLOAT8E4M3FNUZ = 18;
    FLOAT8E5M2 = 19;
    FLOAT8E5M2FNUZ = 20; 

    UINT4 = 21;
    INT4 = 22; 

  }

  // tensor形状.
  repeated int64 dims = 1;

  // tensor类型
  int32 data_type = 2;

  // 对于特别大的张量，将张量分块存储
  message Segment {
    int64 begin = 1;
    int64 end = 2;
  }
  Segment segment = 3;

  // Tensor数据必须按照行优先存储

  // 下面是张量数据类型，这四个字段类型的某个字段出现的话，那上面的data_type必须为对应的这个字段
  repeated float float_data = 4 [packed = true];
  repeated int32 int32_data = 5 [packed = true];
  repeated bytes string_data = 6;
  repeated int64 int64_data = 7 [packed = true];

  // 张量名称
  string name = 8; // namespace Value

  // 模型的描述性字符串
  string doc_string = 12;

  // 某些特别的数据存储方式和位置
  bytes raw_data = 9;
  repeated StringStringEntryProto external_data = 13;
  enum DataLocation {
    DEFAULT = 0;
    EXTERNAL = 1;
  }
  DataLocation data_location = 14;

  repeated double double_data = 10 [packed = true];
  repeated uint64 uint64_data = 11 [packed = true];

  
  repeated StringStringEntryProto metadata_props = 16;
}
```
```
// 稀疏张量数据的格式
message SparseTensorProto {
  TensorProto values = 1;
  TensorProto indices = 2;
  repeated int64 dims = 3;
}
```

```
// 张量形状
message TensorShapeProto {
  message Dimension {
    oneof value {
      int64 dim_value = 1;
      string dim_param = 2;   // 可以是符号变量，nlp中常见，[bat_size, max_seq_len]
    };
    // 维度含义，这个是onnx预定义的标准含义https://github.com/onnx/onnx/blob/main/docs/DimensionDenotation.md#denotation-definition
    string denotation = 3;
  };
  repeated Dimension dim = 1;
}
```

```
// ONNX标准类型, 包括类型，形状
message TypeProto {

  message Tensor {
    int32 elem_type = 1;
    TensorShapeProto shape = 2;
  }

  // 嵌套定义
  message Sequence {
    TypeProto elem_type = 1;
  };

  // 字典
  message Map {
    int32 key_type = 1;
    TypeProto value_type = 2;
  };

  // 可选值类型可以包含所有类型
  message Optional {
    TypeProto elem_type = 1;
  };

  // 稀疏张量
  message SparseTensor {
    int32 elem_type = 1;
    TensorShapeProto shape = 2;
  }

  // 从上面刚定义的几个类型里面选择一个类型
  oneof value {
    Tensor tensor_type = 1;
    Sequence sequence_type = 4;
    Map map_type = 5;
    Optional optional_type = 9;
    SparseTensor sparse_tensor_type = 8;
  }

  // for pre-defined type denotations: https://github.com/onnx/onnx/blob/main/docs/TypeDenotation.md#type-denotation-definition.
  string denotation = 6;
}
```

额外补充说明一下这个几个ONNX和pytorch类型对应关系:
```
Tensor (ONNX) ↔ torch.Tensor (PyTorch)
Sequence (ONNX) ↔ Python 列表 / torch.nn.utils.rnn.PackedSequence (PyTorch)
Map (ONNX) ↔ 字典 dict (PyTorch)
Optional (ONNX) ↔ None (PyTorch)
SparseTensor (ONNX) ↔ torch.sparse.FloatTensor (PyTorch)
```

```
// 规定函数
message FunctionProto {
  // 函数名称，和NodeProto.op_type一样。这个name是在模型中的unique-id (domain, name, overload) of FunctionProtos的一部分
  string name = 1;

  reserved 2;
  reserved "since_version";

  reserved 3;
  reserved "status";

  // 函数的input和output
  repeated string input = 4;
  repeated string output = 5;

  // 函数属性，适用于函数参数没有默认值的情况
  repeated string attribute = 6;

  // The attribute protos of the function.
  // It is for function attributes with default values.
  // A function attribute shall be represented either as
  // a string attribute or an AttributeProto, not both.
  repeated AttributeProto attribute_proto = 11;

  // The nodes in the function.
  repeated NodeProto node = 7;
  // A human-readable documentation for this function. Markdown is allowed.
  string doc_string = 8;

  // The OperatorSets this function body (graph) relies on.
  //
  // All nodes in the function body (graph) will bind against the operator
  // with the same-domain/same-op_type operator with the HIGHEST version
  // in the referenced operator sets. This means at most one version can be relied
  // for one domain.
  //
  // The operator sets imported by FunctionProto should be compatible with the ones
  // imported by ModelProto. Example, if same operator set say 'A' is imported by FunctionProto
  // and ModelProto then versions for the operator set may be different but,
  // the operator schema returned for op_type, domain, version combination
  // for both the versions should be same.

  repeated OperatorSetIdProto opset_import = 9;

  // The domain which this function belongs to.
  // This is part of the unique-id (domain, name, overload) of FunctionProtos in a model.
  string domain = 10;

  // The overload identifier of the function.
  // This is part of the unique-id (domain, name, overload) of FunctionProtos in a model.
  string overload = 13;

  // Information for the values in the function. The ValueInfoProto.name's
  // must be distinct and refer to names in the function (including inputs,
  // outputs, and intermediate values). It is optional for a value to appear
  // in value_info list.
  repeated ValueInfoProto value_info = 12;

  // Named metadata values; keys should be distinct.
  repeated StringStringEntryProto metadata_props = 14;
}
```

docs下的文档很建议看一看，包括算子，类型解释很详细

## 关于`onnx/defs`中的文件是如何进行算子定义的
  - schema.h定义了ONNX_OPERATOR_SET_SCHEMA: 用于注册算子schema
  ```
  #define ONNX_OPERATOR_SET_SCHEMA(name, ver, impl) ONNX_OPERATOR_SET_SCHEMA_EX(name, Onnx, ONNX_DOMAIN, ver, true, impl)
  ......
  #define ONNX_OPERATOR_SET_SCHEMA_EX(name, domain, domain_str, ver, dbg_included_in_static_opset, impl)  \
  class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(domain, ver, name);                                         \
  template <>                                                                                           \
  OpSchema GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(domain, ver, name)>() {                      \
    return impl.SetName(#name).SetDomain(domain_str).SinceVersion(ver).SetLocation(__FILE__, __LINE__); \
  }                                                                                                     \
  size_t dbg_count_check_##name##_##domain##_ver##ver =                                                 \
      (dbg_included_in_static_opset) ? ONNX_DBG_INCREMENT_COUNT_IN_OPSETS() : 0;


  // class OpSchema: 用于管理注册算子, 这个类很长，选取一部分属性展示
  class OpSchema final {
    public:
      static constexpr int kUninitializedSinceVersion = -1; // 未注册版本号，前缀k表示常量，见https://zh-google-styleguide.readthedocs.io/en/latest/google-cpp-styleguide/naming.html#constant-names
    
    // 形参选项配置.
    enum FormalParameterOption : uint8_t {
      Single = 0, // 形参是一个且不可选
      Optional = 1, // 形参是一个且可选
      Variadic = 2, // 可变参数，最少1个
    };

    // 形参是否可微
    enum DifferentiationCategory : uint8_t {
      Unknown = 0,
      Differentiable = 1,
      NonDifferentiable = 2
    };

    // 形参的表示, 包括 input/output name, typeStr, description, and type constraints.
    class FormalParameter final {
      public:
        FormalParameter() = default;

        explicit FormalParameter(
            std::string name,
            DataTypeSet allowed_type_set,
            std::string type_str,
            const std::string& description,
            FormalParameterOption param_option = Single,
            bool is_homogeneous = true,
            int min_arity = 1,
            DifferentiationCategory differentiation_category = Unknown)
            : name_(std::move(name)),
              type_set_(std::move(allowed_type_set)),
              type_str_(std::move(type_str)),
    #ifndef __ONNX_NO_DOC_STRINGS
              description_(description),
    #endif
              param_option_(param_option),
              is_homogeneous_(is_homogeneous),
              min_arity_(min_arity),
              differentiation_category_(differentiation_category) {
    #ifdef __ONNX_NO_DOC_STRINGS
          ONNX_UNUSED_PARAMETER(description);
    #endif
        }

        explicit FormalParameter(
            std::string name,
            const std::string& description,
            std::string type_str,
            FormalParameterOption param_option = Single,
            bool is_homogeneous = true,
            int min_arity = 1,
            DifferentiationCategory differentiation_category = Unknown)
            : name_(std::move(name)),
              type_str_(std::move(type_str)),
    #ifndef __ONNX_NO_DOC_STRINGS
              description_(description),
    #endif
              param_option_(param_option),
              is_homogeneous_(is_homogeneous),
              min_arity_(min_arity),
              differentiation_category_(differentiation_category) {
    #ifdef __ONNX_NO_DOC_STRINGS
          ONNX_UNUSED_PARAMETER(description);
    #endif
        }

        // 返回形参名称
        const std::string& GetName() const;

        // 可以接受的参数类型集合
        const DataTypeSet& GetTypes() const;

        // 形参类型
        const std::string& GetTypeStr() const;

        // 形参描述
        const std::string& GetDescription() const;

        // 形参是 option, Single, Optional or Variadic.
        FormalParameterOption GetOption() const;

        // 形参是否同质，true--同质--所有参数类型相同
        bool GetIsHomogeneous() const;

        // 可变参数情况下，是否是所有
        int GetMinArity() const;

        // 形参可微情况
        DifferentiationCategory GetDifferentiationCategory() const;

      private:
        // 和上面成员函数相应的私有变量
        friend class OpSchema;
        DataTypeSet& MutableTypes();
        std::string name_;
        DataTypeSet type_set_;
        std::string type_str_;
        std::string description_;
        FormalParameterOption param_option_;
        bool is_homogeneous_;
        int min_arity_;
        DifferentiationCategory differentiation_category_;
      };
    ......
    const char* doc() const {
      return doc_.empty() ? nullptr : doc_.c_str();
    ......
    OpSchema& NumInputs(std::set<int> allowed_input_nums);  // 输入参数个数(指定允许接收的多个输入个数的某个数量)

    OpSchema& NumOutputs(std::set<int> allowed_output_nums);  // 输出参数个数

    OpSchema& TypeAndShapeInferenceFunction(InferenceFunction inferenceFunction); // 类型和形状推导函数
    InferenceFunction GetTypeAndShapeInferenceFunction() const {
      return tensor_inference_function_ ? tensor_inference_function_ : dummyInferenceFunction;
    }

    OpSchema& PartialDataPropagationFunction(DataPropagationFunction dataProgationFunction);  // 数据传播函数
    DataPropagationFunction GetDataPropagationFunction() const {
      return data_propagation_function_ ? data_propagation_function_ : dummyDataPropagationFunction;
    }
    ......
    OpSchema& Input(int n, FormalParameter formal_parameter);
    OpSchema& Input(
        int n,
        std::string name,
        const std::string& description,
        std::string type_str,
        FormalParameterOption param_option = Single,
        bool is_homogeneous = true,
        int min_arity = 1,
        DifferentiationCategory differentiation_category = Unknown);

    // Non-STL wrapper to reduce binary size
    OpSchema& Input(
        int n,
        const char* name,
        const char* description,
        const char* type_str,
        FormalParameterOption param_option = Single,
        bool is_homogeneous = true,
        int min_arity = 1,
        DifferentiationCategory differentiation_category = Unknown);

    OpSchema& Output(int n, FormalParameter formal_parameter);

    OpSchema& Output(
        int n,
        std::string name,
        const std::string& description,
        std::string type_str,
        FormalParameterOption param_option = Single,
        bool is_homogeneous = true,
        int min_arity = 1,
        DifferentiationCategory differentiation_category = Unknown);

    // Non-STL wrapper to reduce binary size
    OpSchema& Output(
        int n,
        const char* name,
        const char* description,
        const char* type_str,
        FormalParameterOption param_option = Single,
        bool is_homogeneous = true,
        int min_arity = 1,
        DifferentiationCategory differentiation_category = Unknown);
    }
    ......
  ```
  - 各种对应算子文件夹下的具体算子schemas使用上面定义的ONNX_OPERATOR_SET_SCHEMA, 例如math/def.cc:
  ```
  ONNX_OPERATOR_SET_SCHEMA(
    Add,
    14,
    OpSchema().FillUsing(MathDocGenerator("addition")).PartialDataPropagationFunction([](DataPropagationContext& ctx) {
      MathOpDataPropagator(ctx, "Add");
    }));

  关于FillUsing函数，就是进行一定填充返回，MathDocGenerator产生相关文档和信息填充schema
  OpSchema& OpSchema::FillUsing(const std::function<void(OpSchema&)>& populator) {
    if (populator) {
      populator(*this);
    }
    return *this;
  }
  std::function<void(OpSchema&)> MathDocGenerator(const char* name) {
    return [=](OpSchema& schema) {
      std::string doc;
      POPULATE_OP_DOC_STR(doc = R"DOC(
  Performs element-wise binary {name} (with Numpy-style broadcasting support).

  {broadcast_doc}

  (Opset 14 change): Extend supported types to include uint8, int8, uint16, and int16.
  )DOC";
                          ReplaceAll(doc, "{name}", name);
                          ReplaceAll(doc, "{broadcast_doc}", GenerateBroadcastingDocMul().c_str()););
      schema.SetDoc(doc);
      schema.Input(0, "A", "First operand.", "T", OpSchema::Single, true, 1, OpSchema::Differentiable);
      schema.Input(1, "B", "Second operand.", "T", OpSchema::Single, true, 1, OpSchema::Differentiable);
      schema.Output(
          0,
          "C",
          "Result, has same element type as two inputs",
          "T",
          OpSchema::Single,
          true,
          1,
          OpSchema::Differentiable);
      schema.TypeConstraint(
          "T", OpSchema::all_numeric_types_ir4(), "Constrain input and output types to all numeric tensors.");
      schema.TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
        propagateElemTypeFromInputToOutput(ctx, 0, 0);
        if (hasNInputShapes(ctx, 2))
          bidirectionalBroadcastShapeInference(
              ctx.getInputType(0)->tensor_type().shape(),
              ctx.getInputType(1)->tensor_type().shape(),
              *ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape());
      });
    };
  }
  ```
上面这个函数用到的TypeConstraintParam，其实就是一个允许类型的信息
```
  // Type constraint.
  struct TypeConstraintParam final {
    TypeConstraintParam(
        std::string type_param_str_,
        std::vector<std::string> allowed_type_strs_,
        std::string description_)
        : type_param_str(std::move(type_param_str_)),
          allowed_type_strs(std::move(allowed_type_strs_)),
          description(std::move(description_)) {}

    // Type parameter string, for example, "T", "T1", etc.
    std::string type_param_str;
    // Allowed type strings for <*this> type parameter, for example,
    // "tensor(float)".
    std::vector<std::string> allowed_type_strs;
    // Type parameter description.
    std::string description;
  };
```

看一下OpSchema::TypeAndShapeInferenceFunction这个类型和形状推导函数的实现(仅仅是一个函数的赋值): 
```
OpSchema& OpSchema::TypeAndShapeInferenceFunction(InferenceFunction inferenceFunction) {
  tensor_inference_function_ = std::move(inferenceFunction);
  return *this;
}

// 关于InferenceFunction这个类型代码如下:
// using InferenceFunction = std::function<void(InferenceContext&)>;
```

上面函数调用的InferenceContext在shape_inference.h中，
```
// 类型和形状推导上下文结构体, 输入输出信息
struct InferenceContext {
  virtual const AttributeProto* getAttribute(const std::string& name) const = 0;
  virtual size_t getNumInputs() const = 0;
  virtual const TypeProto* getInputType(size_t index) const = 0;
  virtual bool hasInput(size_t index) const {
    // The default implementation below is used for backward-compatibility
    // for implementations of InferenceContext that don't provide an explicit
    // implementation. This works for normal usage, but may be imprecise in
    // the edge-case where an input is supplied but has no known type.
    // However, inference-methods work only under the assumption that the
    // input-types of all inputs are known.
    return ((index < getNumInputs()) && (getInputType(index) != nullptr));
  }
  virtual const TensorProto* getInputData(size_t index) const = 0;
  virtual size_t getNumOutputs() const = 0;
  virtual TypeProto* getOutputType(size_t index) = 0;
  virtual GraphInferencer* getGraphAttributeInferencer(const std::string& attribute_name) = 0;
  virtual ~InferenceContext() {}
  virtual const SparseTensorProto* getInputSparseData(size_t index) const = 0;
  // Gets the shape inputs computed by partial data propagation.
  virtual const TensorShapeProto* getSymbolicInput(size_t index) const = 0;
};
```

propagateElemTypeFromInputToOutput函数在shape_inference.cc中的实现, getInputType返回的是TypeProto类型，value_case()是protobuf的方法，返回的是proto的message的枚举类型, 它是protobuf编译器在生成代码时自动生成的, 用于处理oneof字段
```
void propagateElemTypeFromInputToOutput(InferenceContext& ctx, size_t inputIndex, size_t outputIndex) {
  auto input_type = ctx.getInputType(inputIndex);
  if (nullptr == input_type) {
    fail_type_inference("Input ", inputIndex, " expected to have type but instead is null");
  }
  const auto input_value_case = input_type->value_case();
  if (input_value_case == TypeProto::kTensorType || input_value_case == TypeProto::kSparseTensorType) {
    propagateElemTypeFromTensorInputToOutput(ctx, inputIndex, outputIndex);
  } else if (input_value_case == TypeProto::kSequenceType) {
    propagateElemTypeFromSequenceInputToOutput(ctx, inputIndex, outputIndex);
  } else if (input_value_case == TypeProto::kOptionalType) {
    propagateElemTypeFromOptionalInputToOutput(ctx, inputIndex, outputIndex);
  } else if (input_value_case == TypeProto::kMapType) {
    propagateElemTypeFromMapInputToOutput(ctx, inputIndex, outputIndex);
  }
}

void propagateElemTypeFromTensorInputToOutput(InferenceContext& ctx, size_t inputIndex, size_t outputIndex) {
  auto input_type = ctx.getInputType(inputIndex);
  if (nullptr == input_type) {
    fail_type_inference("Input type was null");
  }

  const auto input_value_case = input_type->value_case();
  if (input_value_case != TypeProto::kTensorType && input_value_case != TypeProto::kSparseTensorType) {
    fail_type_inference(
        "Input ", inputIndex, " expected to have tensor or sparse tensor type. Got: ", input_value_case);
  }

  const auto input_elem_type = getTensorElementType(*input_type);
  if (input_elem_type == TensorProto::UNDEFINED) {
    fail_type_inference("Element type of input ", inputIndex, " unknown");
  }
  auto output_type = ctx.getOutputType(outputIndex);
  const auto output_value_case = output_type->value_case();
  if (output_value_case == TypeProto::kTensorType || output_value_case == TypeProto::kSparseTensorType) {
    setTensorElementType(input_elem_type, output_value_case, *output_type);
  } else if (output_value_case == TypeProto::VALUE_NOT_SET) {
    // Assume output will have the same type
    setTensorElementType(input_elem_type, input_value_case, *output_type);
  } else {
    // This is not expected to happen
    fail_type_inference(
        "Output ", outputIndex, " expected to have tensor or sparse tensor type. Got: ", output_value_case);
  }
}


inline void setTensorElementType(int32_t elem_type, TypeProto::ValueCase value_case, TypeProto& type) {
  if (value_case == TypeProto::kTensorType) {
    type.mutable_tensor_type()->set_elem_type(elem_type);
  } else if (value_case == TypeProto::kSparseTensorType) {
    type.mutable_sparse_tensor_type()->set_elem_type(elem_type);
  }
}
```


