# OracleGeneral格式支持 - 快速开始

## 概述

项目已完全迁移到OracleGeneral二进制trace格式（支持zstd压缩）。所有算法现在使用高效的24字节/记录二进制格式。

## 快速开始

### 1. 生成测试trace

```bash
cd lib/trace
make test-gen
./generate_test_trace test_trace
```

### 2. 运行算法

```bash
# FOO算法（精确OPT）
cd ../../OHRgoal/FOO
make
./foo ../../lib/trace/test_trace_simple.dat 10000 4 output.txt

# Belady算法
cd ../Belady
make
./belady2 ../../lib/trace/test_trace_simple.dat 10000 100
```

### 3. 运行所有测试

```bash
cd ../..
./test_oracle_general.sh
```

## OracleGeneral格式规范

### 二进制结构（24字节/记录）

```c
struct {
    uint32_t timestamp;         // +0: 请求时间戳 (4字节)
    uint64_t obj_id;            // +4: 对象ID (8字节)
    uint32_t obj_size;          // +12: 对象大小 (4字节)
    int64_t  next_access_vtime; // +16: 下次访问时间 (8字节, -1表示无后续)
}
```

### 文件扩展名

- 无压缩: `.dat`
- 压缩: `.zst` (自动检测并解压)

### 优势

| 特性 | 原文本格式 | OracleGeneral |
|------|-----------|---------------|
| 大小 | ~100字节/请求 | 24字节/请求 |
| 解析速度 | 慢 | 快（二进制） |
| 包含信息 | timestamp, id, size | + next_access_vtime |
| 压缩 | 无 | 可选zstd |

## 测试结果

所有9个算法测试通过：

**OHRgoal (5/5):**
- ✅ FOO - 精确OPT算法
- ✅ PFOO-U - 上界
- ✅ PFOO-L - 下界
- ✅ Belady - 基准
- ✅ Belady-Size - 大小感知

**BHRgoal (3/3):**
- ✅ PFOO-L - BHR下界
- ✅ Belady - BHR基准
- ✅ BeladySplit - 分割对象

**工具 (1/1):**
- ✅ Statistics - 统计工具

## 启用zstd压缩支持（可选）

```bash
# 安装zstd开发库
apt install libzstd-dev

# 在算法目录中编译时启用
make CXXFLAGS+=" -DUSE_ZSTD" LDFLAGS+=" -lzstd"
```

## 核心库文件

- `lib/trace/oracle_general_reader.{h,cpp}` - 二进制trace读取器
- `lib/trace/parse_trace.{h,cpp}` - 统一解析接口
- `lib/trace/generate_test_trace.cpp` - 测试生成器
- `lib/trace/Makefile` - 构建配置

## 故障排除

### 编译错误

**找不到oracle_general_reader.h**
```bash
# 确保Makefile中有
CXXFLAGS += -I ../../lib
```

**链接错误**
```bash
# 确保Makefile的OBJS中有
OBJS += ../../lib/trace/oracle_general_reader.o
```

### 运行时错误

**无法读取trace**
```bash
# 验证文件格式
hexdump -C trace.dat | head
# 应该每条记录24字节
```

**zstd文件读取失败**
```bash
# 需要安装libzstd-dev并重新编译
apt install libzstd-dev
```

## 技术文档

详见各算法目录和`lib/trace/README.md`

