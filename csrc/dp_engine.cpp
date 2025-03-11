#include <Python.h>
#include <vector>
#include <algorithm>
#include <immintrin.h>
#include <xmmintrin.h>
#include <cstring>

// 三维容器约束结构体
struct alignas(64) BinConfig {
    int dims[3];  // 容器三维尺寸 [长, 宽, 高]
};

// 物品数据结构（保持与Python接口兼容）
struct alignas(64) PackItem {
    int orig_dims[3];    // 原始尺寸
    uint8_t valid_rots;  // 有效旋转位掩码
};

// 异常安全的资源管理
struct FreeGuard {
    void* ptr;
    ~FreeGuard() { free(ptr); }
};

// 使用 RAII 管理 Python 列表
class PyListGuard {
    PyObject* list;
public:
    explicit PyListGuard(PyObject* obj = nullptr) : list(obj) {}
    ~PyListGuard() { 
        if (list) {
            Py_DECREF(list); 
        }
    }
    
    // 新增 release 方法
    PyObject* release() noexcept {
        PyObject* ret = list;
        list = nullptr;
        return ret;
    }
    
    // 转换运算符保持原有功能
    operator PyObject*() const { return list; }
    
    // 可选：禁用拷贝（保证资源安全）
    PyListGuard(const PyListGuard&) = delete;
    PyListGuard& operator=(const PyListGuard&) = delete;
};
// SIMD加速的装箱判定（支持三维约束）
inline bool can_place(const int* item_dims, const int* bin_dims) {
    __m128i item = _mm_loadu_si128((const __m128i*)item_dims);
    __m128i space = _mm_loadu_si128((const __m128i*)bin_dims);
    __m128i cmp = _mm_cmpgt_epi32(space, item);
    return (_mm_movemask_epi8(cmp) & 0x777) == 0x777;
}

// 预计算有效旋转方向
constexpr std::array<std::array<int, 3>, 6> rotations = {{
    {0, 1, 2}, {0, 2, 1}, 
    {1, 0, 2}, {1, 2, 0},
    {2, 0, 1}, {2, 1, 0}
}};

// 主装箱算法实现
static PyObject* optimized_3d_packing(PyObject* self, PyObject* args) {
    PyObject* items_list;
    int dim_x, dim_y, dim_z;  // 三维约束参数

    // 解析参数：物品列表 + 三维容量约束
    if (!PyArg_ParseTuple(args, "O(iii)", &items_list, &dim_x, &dim_y, &dim_z)) {
        return nullptr;
    }

    const int num_items = PyList_Size(items_list);
    std::vector<PackItem> items(num_items);
    BinConfig bin_cfg{dim_x, dim_y, dim_z};

    // 预分配内存池
    std::vector<std::vector<int>> bins(1);  // 每个箱子存储物品索引
    int current_bin_remaining[3] = {dim_x, dim_y, dim_z};

    // 读取物品数据并预处理旋转
    for (int i = 0; i < num_items; ++i) {
        PyObject* tuple = PyList_GetItem(items_list, i);
        int w = PyLong_AsLong(PyTuple_GetItem(tuple, 0));
        int h = PyLong_AsLong(PyTuple_GetItem(tuple, 1));
        int d = PyLong_AsLong(PyTuple_GetItem(tuple, 2));
        
        // 预计算有效旋转
        items[i].orig_dims[0] = w;
        items[i].orig_dims[1] = h;
        items[i].orig_dims[2] = d;
        items[i].valid_rots = 0;
        
        for (uint8_t rot = 0; rot < 6; ++rot) {
            const auto& r = rotations[rot];
            if (w <= bin_cfg.dims[r[0]] &&
                h <= bin_cfg.dims[r[1]] &&
                d <= bin_cfg.dims[r[2]]) {
                items[i].valid_rots |= (1 << rot);
            }
        }
    }

    // 按最大维度降序排序（优化空间利用率）
    std::sort(items.begin(), items.end(), [](const PackItem& a, const PackItem& b) {
        return *std::max_element(a.orig_dims, a.orig_dims+3) >
               *std::max_element(b.orig_dims, b.orig_dims+3);
    });

    // 主装箱循环
    for (int i = 0; i < num_items; ++i) {
        bool placed = false;
        int best_rot = -1;
        int temp_remaining[3];

        // 尝试现有箱子（支持旋转）
        for (auto& bin : bins) {
            // 检查所有有效旋转方向
            for (uint8_t rot = 0; rot < 6; ++rot) {
                if (!(items[i].valid_rots & (1 << rot))) continue;
                
                const auto& r = rotations[rot];
                int rotated_dims[3] = {
                    items[i].orig_dims[r[0]],
                    items[i].orig_dims[r[1]],
                    items[i].orig_dims[r[2]]
                };

                // SIMD加速空间检查
                if (can_place(rotated_dims, current_bin_remaining)) {
                    memcpy(temp_remaining, current_bin_remaining, sizeof(temp_remaining));
                    temp_remaining[0] -= rotated_dims[0];
                    temp_remaining[1] -= rotated_dims[1];
                    temp_remaining[2] -= rotated_dims[2];
                    
                    bin.push_back(i);
                    memcpy(current_bin_remaining, temp_remaining, sizeof(temp_remaining));
                    placed = true;
                    best_rot = rot;
                    break;
                }
            }
            if (placed) break;
        }

        // 开新箱
        if (!placed) {
            bins.emplace_back();
            auto& new_bin = bins.back();
            new_bin.push_back(i);
            memcpy(current_bin_remaining, bin_cfg.dims, sizeof(current_bin_remaining));
        }
    }

    // 构建返回结果（返回第一个箱子的物品索引）
    PyListGuard result(PyList_New(0));
    for (int idx : bins[0]) {
        PyList_Append(result, PyLong_FromLong(idx));
    }

    return result.release();
}

// 保持模块结构不变
static PyMethodDef DPMethods[] = {
    {"dp_solver", optimized_3d_packing, METH_VARARGS, "3D bin packing optimizer"},
    {nullptr, nullptr, 0, nullptr}
};

static struct PyModuleDef dpmodule = {
    PyModuleDef_HEAD_INIT,
    "dp_solver",  // 保持模块名称不变
    nullptr,
    -1,
    DPMethods
};

extern "C" PyMODINIT_FUNC PyInit__dp_C() {
    return PyModule_Create(&dpmodule);
}