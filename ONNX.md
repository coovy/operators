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

onnx形状推导:

在onnx中定义有一个枚举, 表示某个维度的值是一个value，还是一个参数，还是未设置:
```
  enum ValueCase {
    kDimValue = 1,
    kDimParam = 2,
    VALUE_NOT_SET = 0,
  };
```
[implementation.h](onnx/onnx/shape_inference)
```
// 用来存储模型中定义的局部函数
using ModelLocalFunctionsMap = std::unordered_map<std::string, const FunctionProto*>;

// 用来存储已知的静态变量的值
using DataValueMap = std::unordered_map<std::string, TensorShapeProto>;

// 使用一个std::unordered_set<std::string> existing_symbols;存储所有参数化的维度
class SymbolTableImpl : public SymbolTable {
 public:
  SymbolTableImpl() : index_(0) {}

  void addFromGraph(const GraphProto& g) override {
    // 返回类型都是ValueInfoProto: 描述张量的元数据和类型信息(name, type, shape)
    AddExistingSymbolicDims(g.input());
    AddExistingSymbolicDims(g.output());
    AddExistingSymbolicDims(g.value_info());
  }
  // Creates a new unique symbol with the given prefix and adds it to the SymbolTable
  // Returns the newly created symbol
  std::string createNew(const std::string& symbol_prefix) override {
    std::string newSymbol;
    do {
      newSymbol = symbol_prefix + std::to_string(index_++);
    } while (existing_symbols.count(newSymbol) > 0);
    existing_symbols.insert(newSymbol);
    return newSymbol;
  }

 private:
  unsigned int index_;
  std::unordered_set<std::string> existing_symbols;

  // TypeProto_Tensor or TypeProto_SparseTensor
  template <typename TensorTypeProto>
  void AddExistingSymbolicDims(const TensorTypeProto& tensorType) {
    if (tensorType.has_shape()) {
      for (int i = 0; i < tensorType.shape().dim_size(); ++i) {
        if (tensorType.shape().dim(i).has_dim_param()) {
          existing_symbols.insert(tensorType.shape().dim(i).dim_param());
        }
      }
    }
  }

  void AddExistingSymbolicDims(const TypeProto& typeProto) {
    const auto val_case = typeProto.value_case();
    switch (val_case) {
      case TypeProto::kTensorType:
        AddExistingSymbolicDims(typeProto.tensor_type());
        break;
      case TypeProto::kSparseTensorType:
        AddExistingSymbolicDims(typeProto.sparse_tensor_type());
        break;
      case TypeProto::kSequenceType:
        AddExistingSymbolicDims(typeProto.sequence_type().elem_type());
        break;
      case TypeProto::kOptionalType:
        AddExistingSymbolicDims(typeProto.optional_type().elem_type());
        break;
      case TypeProto::kMapType:
        AddExistingSymbolicDims(typeProto.map_type().value_type());
        break;
      default:
        break;
    }
  }

  void AddExistingSymbolicDims(const google::protobuf::RepeatedPtrField<ValueInfoProto>& protos) {
    for (const auto& proto : protos) {
      AddExistingSymbolicDims(proto.type());
    }
  }
};


// 存储推导上下文信息，
struct GraphInferenceContext {
  GraphInferenceContext(
      // key: 张量名称， value：外部作用域中值的类型信息
      const std::unordered_map<std::string, TypeProto*>& outer_scope_value_types_by_name_in,
      // 操作集（opset）导入映射，记录了不同域（domain）和版本的对应关系
      const std::unordered_map<std::string, int> opset_imports_in,
      // 符号表
      SymbolTable* symbol_table_in = nullptr,
      // 局部函数unordered_map
      const ModelLocalFunctionsMap& model_local_functions_in = {},
      // 存储opschema的注册表，用于查找和验证opschema
      const ISchemaRegistry* schema_registry_in = OpSchemaRegistry::Instance(),
      DataValueMap* generated_shape_data_by_name_in = nullptr,
      const int ir_version_in = IR_VERSION)
      : outer_scope_value_types_by_name{&outer_scope_value_types_by_name_in},
        opset_imports{opset_imports_in},
        symbol_table{symbol_table_in},
        model_local_functions{model_local_functions_in},
        schema_registry{schema_registry_in},
        generated_shape_data_by_name{generated_shape_data_by_name_in},
        ir_version{ir_version_in} {}

  const std::unordered_map<std::string, TypeProto*>* outer_scope_value_types_by_name;
  const std::unordered_map<std::string, int> opset_imports;
  SymbolTable* symbol_table;
  const ModelLocalFunctionsMap& model_local_functions;
  const ISchemaRegistry* schema_registry;
  DataValueMap* generated_shape_data_by_name;
  const int ir_version;
};
```


```
onnx中的域常量定义
// ONNX domains.
constexpr const char* AI_ONNX_ML_DOMAIN = "ai.onnx.ml";
constexpr const char* AI_ONNX_TRAINING_DOMAIN = "ai.onnx.training";
constexpr const char* AI_ONNX_PREVIEW_TRAINING_DOMAIN = "ai.onnx.preview.training";
// The following two are equivalent in an onnx proto representation.
constexpr const char* ONNX_DOMAIN = "";
constexpr const char* AI_ONNX_DOMAIN = "ai.onnx";

// onnx中的一行代码: using OperatorSetVersion = int;
// 一个unordered_map映射<算子名称, <域名称, <算子版本号, Opschema>>>
// Map type to store operator schemas. The format is,
// <OpName, <Domain, <OperatorSetVersion, OpSchema>>>.
using OpName_Domain_Version_Schema_Map =
    std::unordered_map<std::string, std::unordered_map<std::string, std::map<OperatorSetVersion, OpSchema>>>;
```

```
class ISchemaRegistry {
 public:
  virtual ~ISchemaRegistry() = default;

  virtual const OpSchema*
  GetSchema(const std::string& key, const int maxInclusiveVersion, const std::string& domain = ONNX_DOMAIN) const = 0;
};


class OpSchemaRegistry final : public ISchemaRegistry {
 public:

  // 存储onnx中各个域的op_set的版本信息
  // A singleton class to store domain to min/max op_set version map, as well as
  // domain to last-release op_set version map.
  class DomainToVersionRange final {
   public:
    DomainToVersionRange() {
    }
    ......
    ......
  }
  // 就是获取一个全局的OpName_Domain_Version_Schema_Map，将相应版本的opschema写入这个映射
  class OpSchemaRegisterOnce final {
   public:
    // Export to cpp custom register macro
    OpSchemaRegisterOnce(OpSchema op_schema, int opset_version_to_load = 0, bool fail_duplicate_schema = true) {
      OpSchemaRegisterNoExcept(std::move(op_schema), opset_version_to_load, fail_duplicate_schema);
    }
    static void
    OpSchemaRegisterNoExcept(OpSchema&& op_schema, int opset_version_to_load = 0, bool fail_duplicate_schema = true) {
      ONNX_TRY {
        OpSchemaRegisterImpl(std::move(op_schema), opset_version_to_load, fail_duplicate_schema);
      }
      ONNX_CATCH(const std::exception& e) {
        ONNX_HANDLE_EXCEPTION([&]() { std::cerr << "Schema error: " << e.what() << std::endl; });
      }
    }
    static void
    OpSchemaRegisterImpl(OpSchema&& op_schema, int opset_version_to_load = 0, bool fail_duplicate_schema = true) {
      op_schema.Finalize();
      // 获取一个全局的静态映射表m，用于存储所有操作符的schema信息
      auto& m = GetMapWithoutEnsuringRegistration();
      auto& op_name = op_schema.Name();
      auto& op_domain = op_schema.domain();
      auto& schema_ver_map = m[op_name][op_domain];
      auto ver = op_schema.SinceVersion();
      if (OpSchema::kUninitializedSinceVersion == ver) {
        op_schema.SinceVersion(1);
        ver = op_schema.SinceVersion();
      }

      // Stops because the exact opset_version is registered
      if (schema_ver_map.count(ver)) {
        if (fail_duplicate_schema) {
          const auto& schema = schema_ver_map[ver];
          std::stringstream err;
          err << "Trying to register schema with name " << op_name << " (domain: " << op_domain << " version: " << ver
              << ") from file " << op_schema.file() << " line " << op_schema.line()
              << ", but it is already registered from file " << schema.file() << " line " << schema.line() << std::endl;
          fail_schema(err.str());
        }
        return;
      }

      if (opset_version_to_load != 0) {
        // Stops because the opset_version is higher than opset_version_to_load
        // 确保注册的版本号小于目标版本号，为了兼容性
        if (ver > opset_version_to_load) {
          return;
        }

        // 确保在目标操作集版本范围内没有比当前要注册的操作符版本更高的版本。有就停止
        // Stops because a later version is registered within target opset version
        if (!schema_ver_map.empty()) {
          int max_registered_ver_le_target = GetMaxRegisteredVerWithinTarget(schema_ver_map, opset_version_to_load);
          if (max_registered_ver_le_target >= ver) {
            return;
          }
        }
      }

      CheckDomainAndVersionToRegister(op_schema, op_name, op_domain);
      schema_ver_map.insert(std::pair<int, OpSchema&&>(ver, std::move(op_schema)));
    }

   private:
    // Gets the maximum version from given map that is less or equal to target version
    static int GetMaxRegisteredVerWithinTarget(const std::map<OperatorSetVersion, OpSchema>& m, int target_ver) {
      // std::map is sorted on key
      // reverse iterator returns the largest element keyed on the integer version
      for (auto&& it = m.rbegin(); it != m.rend(); it++) {
        const auto& registered_ver = it->first;
        if (registered_ver <= target_ver) {
          return registered_ver;
        }
      }
      return -1;
    }

    static void CheckDomainAndVersionToRegister(
        const OpSchema& op_schema,
        const std::string& op_name,
        const std::string& op_domain) {
      auto ver_range_map = DomainToVersionRange::Instance().Map();
      auto ver_range_it = ver_range_map.find(op_domain);
      auto ver = op_schema.SinceVersion();

      if (ver_range_it == ver_range_map.end()) {
        std::stringstream err;
        err << "Trying to register schema with name " << op_name << " (domain: " << op_domain << " version: " << ver
            << ") from file " << op_schema.file() << " line " << op_schema.line() << ", but its domain is not"
            << " known by the checker." << std::endl;

        fail_schema(err.str());
      }
      auto lower_bound_incl = ver_range_it->second.first;
      auto upper_bound_incl = ver_range_it->second.second;
      if (!(lower_bound_incl <= ver && upper_bound_incl >= ver)) {
        std::stringstream err;
        err << "Trying to register schema with name " << op_name << " (domain: " << op_domain << " version: " << ver
            << ") from file " << op_schema.file() << " line " << op_schema.line() << ", but its version is not "
            << "in the inclusive range [" << lower_bound_incl << ", " << upper_bound_incl
            << "] (usually, this means you "
            << "bumped the operator version but "
            << "forgot to update the version range in DomainToVersionRange "
            << "in onnx/defs/schema.h)." << std::endl;
        fail_schema(err.str());
      }
    }
  };
  // 简单地从这个unordered_map映射中删除这个opschema
  OpSchemaDeregister(const std::string& op_type, const int version, const std::string& domain = ONNX_DOMAIN) {.....}
  static void OpSchemaDeregisterAll(const std::string& domain = ONNX_DOMAIN) {......}
  // 返回对应的最新版本地schema
  // Return the latest schema for an operator in specified domain.
  // Domain with default value ONNX_DOMAIN means ONNX.
  static const OpSchema* Schema(const std::string& key, const std::string& domain = ONNX_DOMAIN) {}
}
 // 成员变量
 private:
  // OpSchemaRegistry should not need to be instantiated except statically
  // within this class
  // 默认构造私有，单例模式
  OpSchemaRegistry() = default;

  /**
   * @brief Returns the underlying string to OpSchema map.
   *
   * You should not manually manipulate the map object returned. Instead, use
   * the macros defined such as ONNX_OPERATOR_SET_SCHEMA to register your
   * operator schema.
   *
   * We wrap it inside a function to avoid the static initialization order
   * fiasco.
   */
  // 里面就两行代码  static OpName_Domain_Version_Schema_Map map;
  // return map;
  static OpName_Domain_Version_Schema_Map& GetMapWithoutEnsuringRegistration();
  // 内部调用GetMapWithoutEnsuringRegistration()初始化一个静态OpName_Domain_Version_Schema_Map， 然后注册所有opschema
  static OpName_Domain_Version_Schema_Map& map();
  static int loaded_schema_version;
```



```
// 形状推导选项
struct ShapeInferenceOptions {
  // Checks the type-equality for input and output
  bool check_type;
  // 1: Will throw any node level shape infer errors
  // 0: Won't throw node-level shape infer errors, but other errors
  // like merging existing shape with inferred etc are thrown
  int error_mode;
  // Enables data propagation for limited operators
  // to perform shape computation
  bool enable_data_propagation;
  ShapeInferenceOptions(bool check_type_val = false, int strict_mode_val = 0, bool data_prop_val = false)
      : check_type(check_type_val), error_mode(strict_mode_val), enable_data_propagation(data_prop_val){};
};

// 维护一个符号表，用于形状推断
// Maintains a SymbolTable for symbolic shape inference
class SymbolTable {
 public:
  // Adds existing symbols from a main graph or subgraph
  virtual void addFromGraph(const GraphProto& g) = 0;
  // Creates a new symbol which is not duplicate as any existing one
  std::string createNew() {
    return createNew("unk__");
  }
  virtual std::string createNew(const std::string& symbol_prefix) = 0;
  virtual ~SymbolTable() = default;
};

// 在图上执行推理的抽象基类
class GraphInferencer {
 public:
  // Perform inferencing on the graph contained in GraphInferencer.
  // Returns the graph output types post-inferencing.
  virtual std::vector<const TypeProto*> doInferencing(
      const std::vector<const TypeProto*>& inputTypes,
      const std::vector<const TensorProto*>& inputData) = 0;
  virtual ~GraphInferencer() = default;
};

class GraphInferencerImpl : public GraphInferencer {
 public:
  GraphInferencerImpl(GraphProto& g, GraphInferenceContext& context) : g_{&g}, context_{&context}, options_() {}
  GraphInferencerImpl(GraphProto& g, GraphInferenceContext& context, const ShapeInferenceOptions& options)
      : g_{&g}, context_{&context}, options_(options) {}

  std::vector<const TypeProto*> doInferencing(
      const std::vector<const TypeProto*>& inputTypes,
      const std::vector<const TensorProto*>& inputData) override;

 private:
  GraphProto* g_;
  GraphInferenceContext* context_;
  ShapeInferenceOptions options_;
};


// 推理上下文实现
struct InferenceContextImpl : public InferenceContext {
  InferenceContextImpl(
      NodeProto& n,
      const std::unordered_map<std::string, TypeProto*>& valueTypesByName,
      const std::unordered_map<std::string, const TensorProto*>& inputDataByName,
      const std::unordered_map<std::string, const SparseTensorProto*>& inputSparseDataByName,
      const ShapeInferenceOptions& options,
      DataValueMap* generatedShapeData = nullptr,
      GraphInferenceContext* graphInferenceContext = nullptr)
      : graphInferenceContext_{graphInferenceContext}, options_(options), node_(&n) {
    // 获取所有属性存入attributesByName_
    for (auto& attr : *n.mutable_attribute()) {
      attributesByName_[attr.name()] = &attr;
      if (attr.has_g()) {
        // need a mutable GraphProto to run inferencing on this attribute
        graphProtoAttributesByName_[attr.name()] = attr.mutable_g();
      }
    }

    for (const auto& input : n.input()) {
      auto valueTypesIter = valueTypesByName.find(input);
      if (valueTypesIter != valueTypesByName.end()) {
        // 记录所有输入值的类型，放入vector
        allInputTypes_.push_back
        (valueTypesIter->second);
      } else {
        allInputTypes_.push_back(nullptr);
      }

      // input data can be in 1 of the 3 containers
      // inputDataByName - this is when input is TensorProto
      // inputSparseDataByName - this is when input is SparseTensorProto
      // generatedShapeData - this is when input was generated as part of partial data propagation
      // 判断输入数据是普通张量还是稀疏矩阵还是推理过程产生的形状数据，分别存在三个vector
      const auto inputDataIter = inputDataByName.find(input);

      if (inputDataIter != inputDataByName.cend()) {
        allInputData_.push_back(inputDataIter->second);
        allInputSparseData_.push_back(nullptr);
        allShapeInputData_.push_back(nullptr);
      } else {
        allInputData_.push_back(nullptr);
        const auto inputSparseDataIter = inputSparseDataByName.find(input);
        if (inputSparseDataIter != inputSparseDataByName.cend()) {
          allInputSparseData_.push_back(inputSparseDataIter->second);
          allShapeInputData_.push_back(nullptr);
        } else {
          allInputSparseData_.push_back(nullptr);
          if (generatedShapeData != nullptr) {
            const auto inputShapeDataIter = generatedShapeData->find(input);
            if (inputShapeDataIter != generatedShapeData->cend()) {
              allShapeInputData_.push_back(&inputShapeDataIter->second);
            } else {
              allShapeInputData_.push_back(nullptr);
            }
          } else {
            allShapeInputData_.push_back(nullptr);
          }
        }
      }
    }
    // 将输出vector的size大小改变
    allOutputTypes_.resize(n.output_size());
  }

  // 以下是获取各种存储好的数据
  const AttributeProto* getAttribute(const std::string& name) const override {
    ......
  }

  size_t getNumInputs() const override {
    return allInputTypes_.size();
  }

  const TypeProto* getInputType(size_t index) const override {
    ......
  }

  const TensorProto* getInputData(size_t index) const override {
    ......
  }

  const TensorShapeProto* getSymbolicInput(size_t index) const override {
    ......
  }

  const SparseTensorProto* getInputSparseData(size_t index) const override {
    ......
  }

  size_t getNumOutputs() const override {
    return allOutputTypes_.size();
  }

  TypeProto* getOutputType(size_t index) override {
    ......
  }

  // 获取子图的GraphInferencer实例
  GraphInferencer* getGraphAttributeInferencer(const std::string& attr_name) override {
    if (!graphInferenceContext_) {
      fail_type_inference("GraphProto attribute inferencing is not enabled in this InferenceContextImpl instance.");
    }

    GraphInferencer* inferencer = nullptr;

    auto entry = graphAttributeInferencers_.find(attr_name);
    if (entry == graphAttributeInferencers_.cend()) {
      // create GraphInferencer instance
      auto attrNameToGraphProto = graphProtoAttributesByName_.find(attr_name);
      if (attrNameToGraphProto == graphProtoAttributesByName_.cend()) {
        fail_type_inference("Attribute ", attr_name, " does not contain a graph.");
      }

      std::unique_ptr<GraphInferencer> new_inferencer{
          new GraphInferencerImpl(*attrNameToGraphProto->second, *graphInferenceContext_, options_)};

      inferencer = new_inferencer.get();
      graphAttributeInferencers_.emplace(attr_name, std::move(new_inferencer));
    } else {
      inferencer = entry->second.get();
    }

    return inferencer;
  }

  std::string getDisplayName() const override {
    if (node_ == nullptr)
      return "";
    if (node_->domain().empty()) {
      if (node_->name().empty())
        return MakeString("node ", node_->op_type());
      return MakeString("node ", node_->op_type(), " (", node_->name(), ")");
    }
    if (node_->name().empty())
      return MakeString("node ", node_->op_type(), "[", node_->domain(), "]");
    return MakeString("node ", node_->op_type(), "[", node_->domain(), "]", " (", node_->name(), ")");
  }

  std::vector<const TensorProto*> allInputData_;
  std::vector<const SparseTensorProto*> allInputSparseData_;
  std::vector<const TensorShapeProto*> allShapeInputData_;
  std::unordered_map<std::string, const AttributeProto*> attributesByName_;
  std::unordered_map<std::string, GraphProto*> graphProtoAttributesByName_;
  std::vector<const TypeProto*> allInputTypes_;
  std::vector<TypeProto> allOutputTypes_;
  GraphInferenceContext* graphInferenceContext_;

  // mutable as internal cache of GraphInferencer instances
  mutable std::unordered_map<std::string, std::unique_ptr<GraphInferencer>> graphAttributeInferencers_;
  ShapeInferenceOptions options_;
  NodeProto* node_;
};


// 和上面的形状传播上下文类似
struct DataPropagationContextImpl : public DataPropagationContext {
  DataPropagationContextImpl(
      NodeProto& n,
      const std::unordered_map<std::string, TypeProto*>& valueTypesByName,
      const std::unordered_map<std::string, const TensorProto*>& inputDataByName,
      DataValueMap& generatedShapeData)
      : generatedShapeData_(generatedShapeData) {
    size_t input_idx = 0;

    for (auto& attr : *n.mutable_attribute()) {
      // 记录所有属性
      attributesByName_[attr.name()] = &attr;
    }

    for (const auto& input : n.input()) {
       // 将所有输入索引和输入进行映射
      inputIndexToNameMap_.insert({input_idx++, input});

      // 存储所有输入类型
      auto valueTypesIter = valueTypesByName.find(input);
      if (valueTypesIter != valueTypesByName.end()) {
        allInputTypes_.push_back(valueTypesIter->second);
      } else {
        allInputTypes_.push_back(nullptr);
      }

      // 存储所有输入数据
      const auto inputDataIter = inputDataByName.find(input);
      if (inputDataIter != inputDataByName.cend()) {
        allInputData_.push_back(inputDataIter->second);
      } else {
        allInputData_.push_back(nullptr);
      }
    }

    size_t output_idx = 0;
    for (const auto& output : n.output()) {
      outputIndexToNameMap_.insert({output_idx++, output});
    }

    allOutputTypes_.resize(n.output_size());
  }

  const AttributeProto* getAttribute(const std::string& name) const override {
    auto iter = attributesByName_.find(name);
    if (iter == attributesByName_.end()) {
      return nullptr;
    } else {
      return iter->second;
    }
  }

  size_t getNumInputs() const override {
    return allInputTypes_.size();
  }

  const TypeProto* getInputType(size_t index) const override {
    if (index >= allInputTypes_.size()) {
      ONNX_THROW("Input " + ONNX_NAMESPACE::to_string(index) + " is out of bounds.");
    }
    return allInputTypes_[index];
  }

  size_t getNumOutputs() const override {
    return allOutputTypes_.size();
  }

  const TypeProto* getOutputType(size_t index) const override {
    if (index >= allOutputTypes_.size()) {
      ONNX_THROW("Output " + ONNX_NAMESPACE::to_string(index) + " is out of bounds.");
    }
    return &allOutputTypes_[index];
  }

  // Convert integer vector into TensorShapeProto
  template <typename INTEGER>
  void vectorToTensorShapeProto(const std::vector<INTEGER>& input_vals, TensorShapeProto& converted_tsp) const {
    for (unsigned int i = 0; i < input_vals.size(); ++i) {
      converted_tsp.mutable_dim()->Add()->set_dim_value(input_vals[i]);
    }
  }
  // 获取指定索引的输入数据的形状信息
  const TensorShapeProto* getInputData(size_t index) override {
    if (index >= allInputData_.size()) {
      ONNX_THROW("Input " + ONNX_NAMESPACE::to_string(index) + " is out of bounds.");
    }
    const std::string input_name = inputIndexToNameMap_.at(index);
    // Gets it from previous data propagation
    // 如果在生成的形状数据中有形状信息，直接返回
    auto iter = generatedShapeData_.find(input_name);
    if (iter != generatedShapeData_.end()) {
      return &iter->second;
    }
    // Otherwise, gets it from initializer if it exists
    const auto* input_data = allInputData_[index];
    // Only scalar (0D tensor) or 1D tensor can be converted for now
    // TODO: It should support tensors with more dimension on demand
    // 如果在生成的形状数据中没有形状信息，那么解析输入的数据，获取输入数据的维度信息
    if (input_data != nullptr && (input_data->dims_size() == 0 || input_data->dims_size() == 1)) {
      TensorShapeProto tsp;

      if (input_data->data_type() == TensorProto_DataType_INT64) {
        vectorToTensorShapeProto(ParseData<int64_t>(input_data), tsp);
      } else if (input_data->data_type() == TensorProto_DataType_INT32) {
        vectorToTensorShapeProto(ParseData<int32_t>(input_data), tsp);
      } else {
        // Only supports integer type to form a shape
        return nullptr;
      }

      // Adds this TensorShapeProto from initializer into generatedShapeData
      // for future use
      auto result = generatedShapeData_.insert({input_name, std::move(tsp)});
      if (result.second) {
        return &(result.first->second);
      }
    }
    return nullptr;
  }

  // 将输出数据的维度信息存储进generatedShapeData_便于查找
  void addOutputData(size_t index, TensorShapeProto&& tsp) override {
    if (index >= outputIndexToNameMap_.size()) {
      ONNX_THROW("Input " + ONNX_NAMESPACE::to_string(index) + " is out of bounds.");
    }
    auto result = generatedShapeData_.insert({outputIndexToNameMap_.at(index), std::move(tsp)});
    if (!result.second) {
      fail_shape_inference("Data for input  " + ONNX_NAMESPACE::to_string(index) + " already exists.");
    }
  }

  std::vector<const TensorProto*> allInputData_;
  std::unordered_map<size_t, std::string> inputIndexToNameMap_;
  std::unordered_map<size_t, std::string> outputIndexToNameMap_;
  std::vector<const TypeProto*> allInputTypes_;
  std::vector<TypeProto> allOutputTypes_;
  DataValueMap& generatedShapeData_;
  std::unordered_map<std::string, const AttributeProto*> attributesByName_;
};
```
看一看doInferencing这个函数的实现
```

std::vector<const TypeProto*> GraphInferencerImpl::doInferencing(
    const std::vector<const TypeProto*>& input_types,
    const std::vector<const TensorProto*>& input_data) {
  SymbolTable* symbol_table = context_->symbol_table;

  std::cout<<"正在执行doInferencing: "<<std::endl;
  int num_inputs = int(input_types.size());
  std::unordered_set<std::string> initializer_name_set;
  for (const auto& tp : g_->initializer()) {
    initializer_name_set.insert(tp.name());
  }

  if (context_->ir_version >= 4) {
    if (g_->input_size() != num_inputs) {
      fail_shape_inference("Graph has ", g_->input_size(), " inputs but ", num_inputs, " were provided");
    }
    for (int i = 0; i < g_->input_size(); ++i) {
      if (initializer_name_set.count(g_->input(i).name()) > 0) {
        fail_shape_inference(
            "Cannot use the same name as both a subgraph initializer and subgraph input: ", g_->input(i).name());
      }
    }
  } else {
    // IR < 4 requires all initializers to be optional inputs
    // So the number of graph input can be larger than the number of node input
    if (num_inputs > g_->input_size()) {
      fail_shape_inference(
          "Graph has ",
          g_->input_size(),
          " inputs but ",
          num_inputs,
          " were provided.",
          "The number of graph input cannot be smaller than the number of node input");
    } else if (num_inputs < g_->input_size()) {
      for (int i = 0; i < g_->input_size(); ++i) {
        if (i < num_inputs && initializer_name_set.count(g_->input(i).name()) > 0) {
          fail_shape_inference("Graph initializer names must appear after the actual inputs: ", g_->input(i).name());
        } else if (i >= num_inputs && initializer_name_set.count(g_->input(i).name()) == 0) {
          // Further check whether the additional input is in initializers
          fail_shape_inference("Cannot find missing input: ", g_->input(i).name(), "in initializers. ");
        }
      }
    }
  }

  for (int i = 0, end = num_inputs; i < end; ++i) {
    const TypeProto* inferred_input = input_types[i];

    if (!inferred_input)
      continue;

    TypeProto* graph_input = g_->mutable_input(i)->mutable_type();
    // Even if graphInput doesn't have defined type, it will assign inferredType to it
    mergeShapesAndTypes(*inferred_input, graph_input);

    if (symbol_table) {
      MaterializeSymbolicShape(graph_input, *symbol_table);
    }
  }

  // future: pass inputData into InferShapes either directly, or indirectly by
  // updating initializers that match subgraph inputs.
  (void)input_data;
  InferShapesImpl(
      g_,
      // 指向 std::unordered_map<std::string, TypeProto*> 的指针，这个映射用于存储外部作用域中变量的类型信息。在 ONNX 中，这通常用于子图推断时，从外部作用域（即子图的上层图）获取输入类型
      *context_->outer_scope_value_types_by_name, // never null
      context_->opset_imports,
      options_,
      symbol_table,
      context_->model_local_functions,
      context_->schema_registry,
      context_->generated_shape_data_by_name);

  std::vector<const TypeProto*> graph_output_types;
  graph_output_types.reserve(g_->output().size());
  for (const ValueInfoProto& output : g_->output()) {
    graph_output_types.push_back(&output.type());
  }

  return graph_output_types;
}

// 形状推导实现
static void InferShapesImpl(
    GraphProto* g,
    const std::unordered_map<std::string, TypeProto*>& outer_scope_value_types_by_name,
    const std::unordered_map<std::string, int>& opset_imports,
    const ShapeInferenceOptions& options,
    SymbolTable* symbol_table,
    const ModelLocalFunctionsMap& model_local_functions_map,
    const ISchemaRegistry* schema_registry = OpSchemaRegistry::Instance(),
    DataValueMap* generated_shape_data_by_name = nullptr,
    const int ir_version = IR_VERSION // default the latest one
) {
  DataValueMap empty;
  if (generated_shape_data_by_name == nullptr) {
    generated_shape_data_by_name = &empty;
  }
  // 实际调用的类
  ShapeInferenceImplBase base(
      g,
      outer_scope_value_types_by_name,
      opset_imports,
      options,
      symbol_table,
      model_local_functions_map,
      schema_registry,
      generated_shape_data_by_name,
      ir_version);
  base.Process(*g);
  base.FinalizeShapeInference();

// 看内部实际调用的类
class ShapeInferenceImplBase {
 public:
  void UpdateType(const std::string& name, TypeProto* inferred_type) {
    if (inferred_type->value_case() == TypeProto::ValueCase::VALUE_NOT_SET) {
      return;
    }

    if (symbol_table) {
      MaterializeSymbolicShape(inferred_type, *symbol_table);
    }

    // Find any pre-existing type and shape info. If there is such,
    // then check for compatibility with the inferred
    // information. Otherwise, initialize it in an empty state.
    auto iter = value_types_by_name.find(name);
    if (iter != value_types_by_name.end()) {
      mergeShapesAndTypes(*inferred_type, iter->second);
    } else {
      value_types_by_name[name] = inferred_types.Add(name, *inferred_type);
      // For undefined output type, update both value_info and output for now
      // Update existing output with undefined type: assign inferred type to it
      iter = undefined_value_types_by_name.find(name);
      if (iter != undefined_value_types_by_name.end()) {
        *iter->second = *inferred_type;
      }
    }
  }

  void UpdateType(ValueInfoProto& valueInfo) {
    if (valueInfo.has_type()) {
      value_types_by_name[valueInfo.name()] = valueInfo.mutable_type();
    } else {
      undefined_value_types_by_name[valueInfo.name()] = valueInfo.mutable_type();
    }
  }

  template <typename T>
  void AddTemporaryConstant(const std::string& name, const T& vector) {
    input_data_by_name_holder[name] = ToTensor(vector);
    input_data_by_name[name] = &input_data_by_name_holder[name];
  }

  void ProcessConstant(const NodeProto& n) {
    if (IsOnnxDomainOp(n, "Constant") && n.output().size() == 1) {
      const std::string& output_name = n.output(0);
      for (const auto& attr : n.attribute()) {
        if (attr.name() == "value") {
          if (attr.type() == AttributeProto::TENSOR && attr.has_t()) {
            if (reuse_constant_tensors) {
              input_data_by_name[output_name] = &attr.t();
            } else {
              input_data_by_name_holder[output_name] = attr.t();
              input_data_by_name[output_name] = &input_data_by_name_holder[output_name];
            }
          } else if (attr.type() == AttributeProto::SPARSE_TENSOR && attr.has_sparse_tensor()) {
            if (reuse_constant_tensors) {
              input_sparse_data_by_name[output_name] = &attr.sparse_tensor();
            }
          }
        } else {
          switch (attr.type()) {
            case AttributeProto::INTS: {
              std::vector<int64_t> ints{attr.ints().begin(), attr.ints().end()};
              AddTemporaryConstant(output_name, ints);
              break;
            }
            case AttributeProto::INT: {
              std::vector<int64_t> ints({attr.i()});
              AddTemporaryConstant(output_name, ints);
              break;
            }
            case AttributeProto::FLOATS: {
              std::vector<float> floats{attr.floats().begin(), attr.floats().end()};
              AddTemporaryConstant(output_name, floats);
              break;
            }
            case AttributeProto::FLOAT: {
              std::vector<float> floats({attr.f()});
              AddTemporaryConstant(output_name, floats);
              break;
            }
            default:
              break;
          }
        }
      }
    }
  }

  void ProcessCall(const NodeProto& caller, const FunctionProto& callee, InferenceContext& ctx) {
    DataValueMap callee_value_map;
    if (generated_shape_data_by_name != nullptr) {
      BindValuesOnCall(*generated_shape_data_by_name, caller, callee_value_map, callee);
    }
    InferShapeForFunctionNode(
        callee, schema_registry, ctx, options, model_local_functions_map, symbol_table, &callee_value_map);
    if (generated_shape_data_by_name != nullptr) {
      BindValuesOnReturn(callee_value_map, callee, *generated_shape_data_by_name, caller);
    }
  }

  void Process(NodeProto& n) {
    // Resolve domain for node
    auto dit = opset_imports.find(n.domain());
    if (dit == opset_imports.end()) {
      // Both "" (ONNX_DOMAIN) and "ai.onnx" (AI_ONNX_DOMAIN) refer to the default ONNX domain
      if (n.domain() == ONNX_DOMAIN) {
        dit = opset_imports.find(AI_ONNX_DOMAIN);
      }
      if (dit == opset_imports.end()) {
        fail_type_inference(
            "Cannot infer type and shape for node name ",
            n.name(),
            ". No opset import for domain ",
            n.domain(),
            " optype ",
            n.op_type());
      }
    }
    auto domain_version = dit->second;
    const auto schema = schema_registry->GetSchema(n.op_type(), domain_version, n.domain());
    InferenceContextImpl ctx(
        n,
        value_types_by_name,
        input_data_by_name,
        input_sparse_data_by_name,
        options,
        generated_shape_data_by_name,
        &graph_inference_context);

    ONNX_TRY {
      if (schema) {
        if (schema->has_type_and_shape_inference_function()) {
          schema->GetTypeAndShapeInferenceFunction()(ctx);
        } else if (schema->HasFunction()) {
          ProcessCall(n, *(schema->GetFunction()), ctx);
        } // else: rely on schema->CheckInputOutputType() down below.
        // check type-constraints specified via type variables
        if (options.check_type) {
          schema->CheckInputOutputType(ctx);
        }
      } else if (model_local_functions_map.size() > 0) {
        auto iter = model_local_functions_map.find(GetFunctionIdentifier(n));
        if (iter != model_local_functions_map.end()) {
          ProcessCall(n, *(iter->second), ctx);
        } else {
          has_unsupported_op = true;
          return;
        }
      } else {
        has_unsupported_op = true;
        return;
      }
      for (int i = 0; i < n.output_size(); ++i) {
        // skip type and shape propagation for missing optional outputs.
        if (!n.output(i).empty())
          UpdateType(n.output(i), ctx.getOutputType(i));
      }
      // Constant values are tracked to improve inference/checking for subsequent nodes.
      ProcessConstant(n);
      // If data-propagation is enabled, partial-evaluation (aka data-propagation) is performed
      // to improve inference/checking for subsequent nodes.
      if (options.enable_data_propagation && schema && schema->has_data_propagation_function()) {
        if (generated_shape_data_by_name == nullptr) {
          fail_shape_inference(
              "Container for generated shape data cannot be nullptr when enable_data_propagation option is set.");
        }
        DataPropagationContextImpl data_propagation_ctx(
            n, value_types_by_name, input_data_by_name, *generated_shape_data_by_name);
        schema->GetDataPropagationFunction()(data_propagation_ctx);
      }
    }
    ONNX_CATCH(const ONNX_NAMESPACE::InferenceError& ex) {
      ONNX_HANDLE_EXCEPTION([&]() {
        // Note: The following special handling is to accommodate custom-ops. Ideally, custom-ops
        // should be registered with a schema in the schema registry, allowing inference to handle
        // them. As things stand, this special handling is somewhat fragile and is not fully
        // general either. Eg., a custom-op suppresses error-messages for subsequent nodes, but
        // this does not work across graphs. If special handling is required, a user-option may
        // be a better way to do it. The fragility comes from the fact that the types of the
        // returned-values of the custom-op are unknown, and subsequent node-level inference
        // may fail because of this.
        if (!has_unsupported_op) {
          inference_errors.push_back(GetErrorWithNodeInfo(n, ex));
        }
      });
    }
    ONNX_CATCH(const std::runtime_error& err) {
      // TODO: Fix this. Unclear if this should be remapped to a shape inference error.
      // Need to rationalize the different types of exceptions that can be thrown.
      // See: https://github.com/onnx/onnx/pull/5519
      ONNX_HANDLE_EXCEPTION([&]() { fail_shape_inference(GetErrorWithNodeInfo(n, err)); });
    }
  }

  // TypeProto_Tensor or TypeProto_SparseTensor
  template <typename T>
  void ProcessInitializer(
      const std::string& name,
      const T& tensorValue,
      TypeProto& initializer_type,
      std::unordered_map<std::string, const T*>& map) {
    map[name] = &tensorValue;
    auto iter = value_types_by_name.find(name);
    // If it already exists in input, check input and initializer is sync
    // use shape info from input (input has priority over initializer)
    if (iter != value_types_by_name.end()) {
      checkShapesAndTypes(initializer_type, *iter->second);
      // CheckTensorShapesAndTypes(*initializer_tensor_type, *iter->second->mutable_tensor_type());
    }
    // Support IR>=4: some tensors can only exist in initializer and not in input
    // So shape_inference should make use of initializer shapes
    // Store initializer shape info in value_info as well
    else if (ir_version >= 4) {
      initializer_type_list.push_back(std::move(initializer_type));
      value_types_by_name[name] = &initializer_type_list.back();
    }
  }

  void Process(GraphProto& graph) {
    if (symbol_table) {
      TraverseGraphsToAddExistingSymbols(graph, *symbol_table);
    }
    for (auto& vi : *graph.mutable_value_info()) {
      UpdateType(vi);
    }
    for (auto& vi : *graph.mutable_input()) {
      UpdateType(vi);
    }
    for (auto& vi : *graph.mutable_output()) {
      UpdateType(vi);
    }
    for (const auto& tp : graph.initializer()) {
      TypeProto initializer_type;
      TypeProto_Tensor* initializer_tensor_type = initializer_type.mutable_tensor_type();
      initializer_tensor_type->set_elem_type(tp.data_type());
      // set the shape according to the initializer shape info
      auto* shape = initializer_tensor_type->mutable_shape();
      for (int i = 0; i < tp.dims_size(); ++i) {
        shape->add_dim()->set_dim_value(tp.dims(i));
      }
      ProcessInitializer(tp.name(), tp, initializer_type, input_data_by_name);
    }
    for (const auto& tp : graph.sparse_initializer()) {
      TypeProto initializer_type;
      auto* initializer_sparse_tensor_type = initializer_type.mutable_sparse_tensor_type();
      initializer_sparse_tensor_type->set_elem_type(tp.values().data_type());
      // set the shape according to the initializer shape info
      auto* shape = initializer_sparse_tensor_type->mutable_shape();
      for (int i = 0; i < tp.dims_size(); ++i) {
        shape->add_dim()->set_dim_value(tp.dims(i));
      }
      ProcessInitializer(tp.values().name(), tp, initializer_type, input_sparse_data_by_name);
    }
    for (auto& n : *graph.mutable_node()) {
      Process(n);
    }
  }

  void Process(const NodeProto& n, internal::AttributeBinder& attribute_binder) {
    NodeProto copy_n(n);
    attribute_binder.VisitNode(&copy_n);
    Process(copy_n);
  }

  void Process(const FunctionProto& func_proto, InferenceContext& ctx) {
    // Ensure Constant node tensor-attributes are copied
    bool old_reuse_constant_tensors = reuse_constant_tensors;
    reuse_constant_tensors = false;

    // Get a temporary tensor-shape map
    const int num_actual_inputs = static_cast<int>(ctx.getNumInputs());
    const auto num_func_inputs = func_proto.input_size();
    std::vector<TypeProto> types_cache(num_func_inputs);
    for (int i = 0; i < num_func_inputs; ++i) {
      auto& parameter_name = func_proto.input().Get(i);
      auto* type_ptr = (i < num_actual_inputs) ? ctx.getInputType(i) : nullptr;
      // nullptr is valid, and indicates a missing optional input
      if (type_ptr != nullptr) {
        // Use a temporary copy of original type.
        // TODO: investigate whether we can eliminate use of temporary copy
        types_cache[i] = *type_ptr;
        value_types_by_name[parameter_name] = &types_cache[i];
      } else
        value_types_by_name[parameter_name] = nullptr;
    }

    // Create a temporary initializer value map
    for (int i = 0; i < num_actual_inputs && i < num_func_inputs; ++i) {
      const TypeProto* type = ctx.getInputType(i);
      if (type != nullptr) {
        if (type->value_case() == TypeProto::kTensorType && ctx.getInputData(i) != nullptr) {
          input_data_by_name[func_proto.input().Get(i)] = ctx.getInputData(i);
        } else if (type->value_case() == TypeProto::kSparseTensorType && ctx.getInputSparseData(i) != nullptr) {
          input_sparse_data_by_name[func_proto.input().Get(i)] = ctx.getInputSparseData(i);
        }
      }
    }

    std::unordered_map<std::string, const AttributeProto*> attr_map;
    for (auto& attr : func_proto.attribute()) {
      if (ctx.getAttribute(attr) != nullptr) {
        attr_map[attr] = ctx.getAttribute(attr);
      }
    }

    for (auto& default_value : func_proto.attribute_proto()) {
      const std::string& name = default_value.name();
      const AttributeProto* value = ctx.getAttribute(name);
      attr_map[name] = (value != nullptr) ? value : &default_value;
    }

    internal::AttributeBinder attribute_binder(attr_map);
    for (auto& n : func_proto.node()) {
      Process(n, attribute_binder);
    }

    for (int i = 0; i < func_proto.output_size(); ++i) {
      const std::string& output_name = func_proto.output().Get(i);
      // Skip if no type inferred for the tensor
      auto iter = value_types_by_name.find(output_name);
      if (iter != value_types_by_name.cend()) {
        // Copy the type info to ctx
        // to pass back to main graph
        auto type_proto = ctx.getOutputType(i);
        type_proto->CopyFrom(*(iter->second));
      }
    }

    reuse_constant_tensors = old_reuse_constant_tensors;
  }

 public:
  ShapeInferenceImplBase(
      GraphProto* graph, // nullptr for FunctionProto inference
      const std::unordered_map<std::string, TypeProto*>& outer_scope_value_types_by_name_in,
      const std::unordered_map<std::string, int>& opset_imports_in,
      const ShapeInferenceOptions& options_in,
      SymbolTable* symbol_table_in,
      const ModelLocalFunctionsMap& model_local_functions_map_in,
      const ISchemaRegistry* schema_registry_in = OpSchemaRegistry::Instance(),
      DataValueMap* generated_shape_data_by_name_in = nullptr,
      const int ir_version_in = IR_VERSION // default the latest one
      )
      : inferred_types(graph),
        value_types_by_name(outer_scope_value_types_by_name_in),
        opset_imports(opset_imports_in),
        options(options_in),
        symbol_table(symbol_table_in),
        model_local_functions_map(model_local_functions_map_in),
        schema_registry(schema_registry_in),
        generated_shape_data_by_name(generated_shape_data_by_name_in),
        ir_version(ir_version_in),
        graph_inference_context{
            value_types_by_name,
            opset_imports,
            symbol_table,
            model_local_functions_map,
            schema_registry,
            generated_shape_data_by_name,
            ir_version} {
    if (options.enable_data_propagation && generated_shape_data_by_name == nullptr) {
      fail_shape_inference(
          "Container for generated shape data cannot be nullptr when enable_data_propagation option is set.");
    }
  }

  void FinalizeShapeInference() {
    auto& errors = getErrors();
    // Throw shape inference error if any. Error mode right now only supports 0 and 1.
    // When set to 0, any node level shape inference errors are not thrown. This is to support backward compatiblity
    // with 1.7 and earlier releases. When set to 1 it will throw all exceptions.
    // TODO: Add a more granular way for exception handling.
    if (!errors.empty() && options.error_mode > 0) {
      std::string full_errors = "Inference error(s): ";
      for (const std::string& error : inference_errors) {
        full_errors += error + "\n";
      }
      fail_shape_inference(full_errors);
    }
  }

  const std::vector<std::string>& getErrors() const {
    return inference_errors;
  }

 private:
  InferredTypes inferred_types;
  std::unordered_map<std::string, TypeProto*> value_types_by_name;
  const std::unordered_map<std::string, int>& opset_imports;

  const ShapeInferenceOptions& options;
  SymbolTable* symbol_table;
  const ModelLocalFunctionsMap& model_local_functions_map;
  const ISchemaRegistry* schema_registry;
  DataValueMap* generated_shape_data_by_name;
  int ir_version;
  GraphInferenceContext graph_inference_context;

  std::unordered_map<std::string, TypeProto*> undefined_value_types_by_name;
  std::unordered_map<std::string, const TensorProto*> input_data_by_name;
  std::unordered_map<std::string, TensorProto> input_data_by_name_holder;
  std::unordered_map<std::string, const SparseTensorProto*> input_sparse_data_by_name;

  bool has_unsupported_op = false;

  std::vector<std::string> inference_errors;

  std::list<TypeProto> initializer_type_list;

  // reuse_constant_tensors: controls whether we need to copy tensors occurring as attributes
  // in Constant nodes. We avoid it for inference for graphs, but must make a copy for functions.
  bool reuse_constant_tensors = true;
};

}
```




