---
name: ocarina
description: "Develop in the Ocarina GPU shading DSL framework. Use when: writing/modifying DSL kernels, Var/Expr/Ref types, Encodable serialization, AST nodes, code generation, RHI device commands, backend (CUDA/Vulkan) code, framework rendering primitives, or any code under src/ocarina/. Covers architecture, type system, coding conventions, known pitfalls, and build system."
---

# Ocarina Development Skill

Before writing or modifying any ocarina code, review the relevant section in `docs/README.md` within this skill folder. The full architecture, conventions, DSL patterns, pitfalls, and build commands are documented there.

## Quick Reference

### Module Map (dependency order)

```
ocarina-ext        → third-party libs (stb, tinyexr, xxhash, spdlog, EASTL, fmt)
ocarina-core       → platform, memory, threading, logging, image, hash
ocarina-ast        → Type, Expression, Statement, Variable, Function
ocarina-generator  → CppCodegen (AST → C++ source text)
ocarina-rhi        → Device, Stream, Buffer, Texture, Command abstractions
ocarina-dsl        → Var<T>, Expr<T>, Ref<T>, Kernel, Callable, Encodable, operators
ocarina-backend-*  → CUDA (+ OptiX), Vulkan (MODULE plugins, runtime-loaded)
ocarina-framework  → Camera, Primitive, Renderer, Material (high-level scene)
ocarina-GUI        → Window, Widget (SHARED)
ocarina-GUI_impl   → SDL2 + ImGui plugin (MODULE)
```

### Critical Rules

1. **Encodable const**: `host_value_` and `device_value_` are NOT mutable; use `const_cast` in `decode()` — makes const violations visible in review.
2. **Swizzle write**: Never modify `expr_value_t<>` or `extract_expression()`; use `remove_device_t<T>` in concepts, call `Swizzle::decay()` → `Var<vec_type>` then recurse.
3. **DLL export**: Each module has its own `OC_*_API` macro (`OC_CORE_API`, `OC_AST_API`, `OC_DSL_API`, etc.). Always decorate exported symbols.
4. **Namespace**: Primary `ocarina`, inline sub-namespace `ocarina::inline core`. Never use `using namespace ocarina` in headers.
5. **Type registration**: `Type::of<T>()` → `TypeDesc<T>::description()` → `TypeRegistry::parse_type()`. Add `TypeDesc` specialization for new types.
6. **Function context**: DSL code executes inside `Function::push/pop` stack. `Function::current()` returns active context. Never call DSL operators outside a function context.
7. **Header style**: `#pragma once`, include order: stdlib → external → internal.

### Build

```powershell
# From cmake-build-debug or cmake-build-release
cmd /c '"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" x64 >nul 2>&1 && ninja -j8 <target> 2>&1'
```

Key targets: `ocarina-core`, `ocarina-ast`, `ocarina-dsl`, `ocarina-backend-cuda`, `test-callable`, `test-kernel`, `test-soa`

### DSL Pattern Cheatsheet

```cpp
// Define a callable function
Callable<float(float, float)> add = [](Var<float> a, Var<float> b) {
    return a + b;
};

// Define a kernel
Kernel<void(Buffer<float>)> kern = [&](BufferVar<float> buf) {
    Var<uint> tid = thread_id();
    buf.write(tid, add(buf.read(tid), 1.f));
};

// Control flow
$if(x > 0) { y = x; } $elif(x < 0) { y = -x; } $else { y = 0.f; };
$for(i, range(n)) { sum += arr.read(i); };
$loop { $if(cond) { $break; }; };

// Launch
stream << device.compile(kern)(buffer).dispatch(N) << synchronize() << commit();
```
