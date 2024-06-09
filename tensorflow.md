# 关于tensorflow的内存管理

[TensorUsageRecord](tensorflow\tensorflow\lite\delegates\gpu\common\memory_management\types.h)
```
// 记录tensor size, 第一个与最后一个使用它的task(一个task代表一个算子操作)(代表生命周期)
template <typename TensorSizeT>
struct TensorUsageRecord {
  TensorSizeT tensor_size;    // tensor size
  TaskId first_task;          // 第一个算子操作
  TaskId last_task;           // 最后一个算子操作

  TensorUsageRecord(TensorSizeT size, TaskId first, TaskId last)
      : tensor_size(size), first_task(first), last_task(last) {}

  // Default order of tensor usage records is increasing order of first_task.
  bool operator<(const TensorUsageRecord<TensorSizeT>& other) const {
    return first_task < other.first_task;
  }
};


// ObjectsAssignment用于存储内存分配信息，包含内存块id的数组、内存块的size的数组，他们的下标索引都是tensor的id
// Information about assignment of tensors to shared objects
template <typename TensorSizeT>
struct ObjectsAssignment {
  // shared_object_ids_[i] is ID of shared object, that tensor i will be using.
  std::vector<size_t> object_ids;
  // shared_object_sizes_[i] is a size of shared object with ID equal to i.
  std::vector<TensorSizeT> object_sizes;
};

// OffsetsAssignment用于记录一块大内存的中的分配信息
struct OffsetsAssignment {
  std::vector<size_t> offsets;
  size_t total_size;
};
```
见测试用例[TEST(Model, ChainRecords)](tensorflow\tensorflow\lite\delegates\gpu\common\memory_management_test.cc)
对应的OneRecord,ChainRecords, ComplexRecords测试用例，和多种分配策略的测试
```
// ChainRecords例子，可以看到关键的函数AssignObjectsToTensors
TEST(Model, ChainRecords) {
  std::vector<TensorUsageRecord<size_t>> usage_records{
      {/*size=*/16, /*first=*/0, /*last=*/1},
      {/*size=*/8, /*first=*/1, /*last=*/2},
      {/*size=*/64, /*first=*/2, /*last=*/3},
      {/*size=*/32, /*first=*/3, /*last=*/4},
      {/*size=*/8, /*first=*/4, /*last=*/5},
  };

  ObjectsAssignment<size_t> assignment;
  ASSERT_TRUE(
      AssignObjectsToTensors(usage_records, MemoryStrategy::NAIVE, &assignment)
          .ok());
    ......
```
```
// 多种内存分配策略
// Calculates the assignment of shared objects to given tensors, including
// objects' sizes. Below there are specializations for different types, that
// support more memory strategies.
// If reallocation_graph is provided, assignment of shared objects support
// parallel order of operation execution, but memory consumption in this case
// can be larger. Currently only GREEDY_IN_ORDER strategy can use this
// reallocation_graph.
template <typename TensorSizeT>
absl::Status AssignObjectsToTensors(
    const std::vector<TensorUsageRecord<TensorSizeT>>& usage_records,
    MemoryStrategy strategy, ObjectsAssignment<TensorSizeT>* assignment,
    const UsageGraph* reallocation_graph = nullptr) {
  switch (strategy) {
    case MemoryStrategy::NAIVE:
      return NaiveAssignment(usage_records, assignment);
    case MemoryStrategy::EQUALITY:
      return EqualityAssignment(usage_records, assignment);
    default:
      return absl::InternalError(
          "MemoryStrategy is not supported with current tensor size type.");
  }
  return absl::OkStatus();
}

// 不同的分配策略
enum class MemoryStrategy {
  // Naive strategy is to allocate each object separately.
  // Can be useful for debugging to see all intermediate outputs.
  NAIVE,

  // Equality strategy allows to reuse the same part of memory for several
  // tensors with the same size, but non-intersecting usage intervals.
  EQUALITY,

  // Greedy strategy uses greedy algorithm, iterating through all the tensors in
  // order of their first_task, to reuse memory from tensors, that
  // won't be used anymore, for new ones.
  GREEDY_IN_ORDER,

  // Greedy by size strategy uses greedy algorithm, iterating through all the
  // tasks in non-increasing of their breadth, and calculating allocations for
  // tensors used in these tasks. By breadth of the task we understand sum of
  // sizes of all tensors in its TaskProfile.
  GREEDY_BY_BREADTH,

  // Greedy by size strategy uses greedy algorithm, iterating through all the
  // tensors in non-increasing of their size, to reuse memory from tensors, that
  // won't be used anymore, for new ones.
  GREEDY_BY_SIZE,

  // Choose greedy strategy from several fast algorithms, that provides best
  // memory allocation for the given usage records.
  GREEDY_BEST,

  // Mincostflow strategy consists of building auxiliary flow graph and solving
  // the minimum-cost flow problem in it. In the end edges with zero residual
  // capacity determine assignment of shared objects to tensors.
  MINCOSTFLOW,
};
```

然后是Equality策略的实现，通过生命周期管理内存使用，
EqualityAssignmentWithHash传入参数包含所有tensor生命周期的数组usage_records和存储分配结果的assignment; <br />
然后执行过程:<br />
初始化一个pool为map类型，key: 内存块的size, value: 空闲共享内存块的id的数组<br />
初始化一个objects_in_use为优先队列priority_queue, 数据类型是一个结构体QueueRecord, 包含两个成员: 内存块id和使用这个内存块的tensor的last_task
```
struct QueueRecord {
  QueueRecord(TaskId task_id, size_t obj_id)
      : last_task(task_id), object_id(obj_id) {}

  // Objects in queue are ordered by last_task.
  bool operator<(const QueueRecord& other) const {
    return (last_task > other.last_task) ||
           (last_task == other.last_task && object_id > other.object_id);
  }

  // Last task, where shared object is used.
  TaskId last_task;
  size_t object_id;
};
```
然后遍历所有tensor进行操作: 首先每遍历到一个tensor, 通过比较当前tensor的first_task和大根堆objects_in_use的top元素的last_first, 
如果objects_in_use的top元素占据的内存块不再使用就将这个内存块放进空闲内存块池pool，
然后判断如果空闲内存块池中没有size等于当前tensor的size的内存块，那就新建一个内存块，大小为tensor的size,并且加入到objects_in_use，
如果找到相等size的内存块，那就取出一块给当前tensor，
遍历结束，分配结束。

然后还提供了另外一种实现:
使用了vector数组dealloc_task，它的索引是内存块的id, value是使用当前内存块的tensor的last_task,
所以它的实现过程也是遍历所有tensor，更新存储分配结果的assignment, 使用了vector数组dealloc_task这些。
```
template <typename TensorSizeT>
absl::Status EqualityAssignmentWithHash(
    const std::vector<TensorUsageRecord<TensorSizeT>>& usage_records,
    ObjectsAssignment<TensorSizeT>* assignment) {
  size_t num_records = usage_records.size();
  assignment->object_sizes.clear();
  assignment->object_ids.assign(num_records, kNotAssigned);

  // Pool is a map with size as a key and vector with ids of free shared objects
  // of this size as a value.
  absl::flat_hash_map<TensorSizeT, std::vector<size_t>> pool;
  std::priority_queue<QueueRecord> objects_in_use;
  for (size_t i = 0; i < num_records; ++i) {
    // Pop from the queue and add to the pool all objects that are no longer
    // in use at the time of execution of the first_task of i-th intermediate
    // tensor.
    while (!objects_in_use.empty() &&
           objects_in_use.top().last_task < usage_records[i].first_task) {
      auto object_id = objects_in_use.top().object_id;
      pool[assignment->object_sizes[object_id]].push_back(object_id);
      objects_in_use.pop();
    }

    const TensorSizeT tensor_size = usage_records[i].tensor_size;
    auto pool_it = pool.find(tensor_size);
    if (pool_it == pool.end() || pool_it->second.empty()) {
      // No free shared object with size equal to tensor_size. Create a new one,
      // assign i-th tensor to it and add to the queue of objects in use.
      assignment->object_ids[i] = assignment->object_sizes.size();
      assignment->object_sizes.push_back(tensor_size);
      objects_in_use.push(
          {usage_records[i].last_task, assignment->object_ids[i]});
    } else {
      // Shared object with id it->second has size equal to tensor_size. Reuse
      // this object: erase it from pool and add to the queue of objects in use.
      assignment->object_ids[i] = pool_it->second.back();
      pool_it->second.pop_back();
      objects_in_use.push(
          {usage_records[i].last_task, assignment->object_ids[i]});
    }
  }
  return absl::OkStatus();
}
```
可以看看这两篇链接:
https://discuss.tf.wiki/t/topic/872
https://blog.csdn.net/jinzhuojun/article/details/128979978?spm=1001.2014.3001.5506
这两篇论文:
Efficient Memory Management for Deep Neural Net Inference
On-Device Neural Net Inference with Mobile GPUs
还是详细再总结一下:


