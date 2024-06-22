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
end-----这样就是从时序的角度去分配
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

然后是看`greedy_by_breadth_assignment.cc`的实现，这个是比较巧妙的一种思路
总体思路: 每个task运行时间，此时肯定存在一些中间张量，所以用一个task_profiles记录每个task运行时会存在的所有中间张量，task_profiles的结构就是二维vector
然后task_breadth存储每个task，和对应的所有中间张量的size之和，按照size从大到小排序，
然后obj_schedules存储着所有已经分配的共享对象(buffer)，和每个共享对象现在已经分配的tensor，所以obj_schedules是一个vector,元素是共享同一个buffer的tensor集合
然后就是遍历task_breadth，从拥有最大中间张量size的task开始，再遍历它的中间张量，去和obj_schedules中的buffer和对应的tensor对比，选取可以使用的buffer，
如果有多个可以使用就用当前tensor的size和buffer的size差别最合适的,没有就新建buffer。
end------这样就是从空间的角度去贪心分配
```

// 传入一个包含所有tensor使用记录的vector: usage_records
absl::Status GreedyByBreadthAssignment(
    const std::vector<TensorUsageRecord<size_t>>& usage_records,
    ObjectsAssignment<size_t>* assignment) {
  std::vector<TaskProfile> task_profiles = CalculateTaskProfiles(usage_records);
  // task_profile是一个vector，下标代表task_id, 每个元素是一个vector，包含了task_id对应的所有tensor使用记录, 所以task_profile的含义是存储相应task_id执行时所有的tensor

  // Task breadth is a sum of sizes of all tensors in its TaskProfile
  std::vector<TaskBreadthWithId> task_breadth;  // task_breadth代表每个task的宽度
  for (size_t task_id = 0; task_id < task_profiles.size(); ++task_id) {
    size_t breadth = 0;
    for (const auto& tensor_info : task_profiles[task_id]) {
      breadth += tensor_info.usage_record->tensor_size;
    }
    task_breadth.emplace_back(breadth, task_id);
  }

  assignment->object_sizes.clear();
  assignment->object_ids.assign(usage_records.size(), kNotAssigned);
  std::vector<SharedObjectSchedule> obj_schedules;    // Set of usage records for all tensors assigned to the shared object, ordered by first_task.

  // Iterate through all tasks in non-increasing order of their breadth.
  std::stable_sort(task_breadth.rbegin(), task_breadth.rend());
  for (const auto& task : task_breadth) {
    // Iterate through all tensors, that must be allocated during the execution
    // of task, in non-increasing order of their tensor_size.
    for (const auto& tensor_info : task_profiles[task.task_id]) {
      // 遍历思路：从宽度最大的task开始 -> 找到task_profiles的所有中间张量， 然后从宽度最小的tensor开始，tensor_info就是一个vector<TensorUsageWithIndex>
      if (assignment->object_ids[tensor_info.idx] != kNotAssigned) {
        // 如果当前tensor已经被分配了，那么就跳过
        continue;
      }
      const auto& rec = *tensor_info.usage_record; // rec是一个TensorUsageRecord结构体，现在要为rec分配object
      const size_t num_objects = obj_schedules.size();  // obj_schedules是一个vector，每个元素是一个set, 代表一个object的所有tensor使用记录, 按照first排序
      size_t best_object = num_objects;
      for (size_t obj_id = 0; obj_id < num_objects; ++obj_id) {
        // 为这个TensorUsageRecord找到一个最合适的object，可能有多个object合适，哪个最合适？遍历所有object，某个可用的object且这个object的tensor_size和rec的tensor_size最接近
        // If size of current_object is worse than size of best found before, we
        // can skip it.
        if (best_object != num_objects) {
          const size_t best_size = assignment->object_sizes[best_object];
          const size_t cur_size = assignment->object_sizes[obj_id];
          if (best_size < rec.tensor_size) {
            if (cur_size <= best_size) {
              // best_size is smaller than tensor_size, but cur_size is even
              // smaller.
              continue;
            }
          } else if (cur_size < rec.tensor_size || cur_size >= best_size) {
            // best_size is larger or equal to tensor_size, and cur_size is
            // either smaller than tensor_size, or too large.
            continue;
          }
        }
        const auto& schedule = obj_schedules[obj_id]; // schedule是一个set，存储了obj_id这个object的所有tensor使用记录
        auto it = schedule.lower_bound(rec); // 返回第一个first大于等于rec的first的元素
        bool update_best_object = true;
        if (it != schedule.end() && it->first_task <= rec.last_task) {
          // 如果找到了一个first大于等于rec的first的元素，且这个元素的first_task小于等于rec的last_task，那就说明这两个tensor的使用时间有重叠，不能复用同一个object
          // Some tensor, which usage interval intersects with current, already
          // assigned to this object.
          update_best_object = false;
        }
        if (update_best_object && it != schedule.begin()) {
          it--;
          if (it->last_task >= rec.first_task) {
            // 说明和前一个tensor的使用时间有重叠，不能复用同一个object
            // Some tensor, which usage interval intersects with current,
            // already assigned to this object.
            update_best_object = false;
          }
        }
        if (update_best_object) {
          // 说明和前后的tensor都没有重叠，可以复用同一个object
          best_object = obj_id;
        }
      }
      if (best_object == num_objects) {
        // Create new shared object and assign current tensor to it.
        obj_schedules.push_back({rec});
        assignment->object_sizes.push_back(rec.tensor_size);
      } else {
        // Assign current tensor to best_object.
        obj_schedules[best_object].insert(rec);
        // Size of best_object can be increased, if it is smaller than
        // tensor_size.
        assignment->object_sizes[best_object] =
            std::max(assignment->object_sizes[best_object], rec.tensor_size);
      }
      assignment->object_ids[tensor_info.idx] = best_object;
    }
  }
  // In the end all tensors must be assigned to some objects.
  for (const auto& obj_id : assignment->object_ids) {
    if (obj_id == kNotAssigned) {
      return absl::InternalError("Error while calculating the assignment.");
    }
  }
  return absl::OkStatus();
}

```


