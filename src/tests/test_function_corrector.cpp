//
// Created by GitHub Copilot on 2026/04/05.
//

#include "dsl/dsl.h"
#include "dsl/func.h"
#include "ast/function.h"
#include "ast/expression.h"

#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

using namespace ocarina;

namespace {

void fail(const std::string &message) {
    std::cerr << "[test_function_corrector] " << message << std::endl;
    std::exit(EXIT_FAILURE);
}

void require(bool condition, const std::string &message) {
    if (!condition) {
        fail(message);
    }
}

template<typename T>
void require_eq(const T &actual, const T &expected, const std::string &message) {
    if (!(actual == expected)) {
        fail(message + " expected=" + std::to_string(expected) + " actual=" + std::to_string(actual));
    }
}

const CallExpr *require_call_expr(const shared_ptr<Function> &function, const std::string &name) {
    require(function != nullptr, name + " function was not captured");
    const CallExpr *call_expr = function->current_call_expr();
    require(call_expr != nullptr, name + " call expression is null");
    return call_expr;
}

std::vector<const Function *> collect_custom_functions(const shared_ptr<Function> &function) {
    require(function != nullptr, "owner function is null");
    std::vector<const Function *> result;
    function->for_each_custom_func([&](const Function *custom_function) {
        result.push_back(custom_function);
    });
    return result;
}

const Function *require_single_custom_function(const shared_ptr<Function> &function, const std::string &name) {
    std::vector<const Function *> custom_functions = collect_custom_functions(function);
    require_eq(custom_functions.size(), static_cast<size_t>(1), name + " custom function count mismatch");
    return custom_functions.front();
}

void test_single_capture_deduplicates_repeated_use() {
    shared_ptr<Function> kernel_function;

    [[maybe_unused]] Kernel kernel = [&] {
        Var<int> outer = 7;
        Lambda capture = [&](Var<int> value) {
            return value + outer + outer;
        };
        [[maybe_unused]] Var<int> result = capture(3);
    };
    kernel_function = kernel.function();

    const Function *callable_function = require_single_custom_function(kernel_function, "single capture");
    require_eq(callable_function->arguments().size(), static_cast<size_t>(0), "single capture explicit arg count mismatch");
    require_eq(callable_function->appended_arguments().size(), static_cast<size_t>(2), "single capture appended arg count mismatch");
    require_eq(callable_function->all_arguments().size(), static_cast<size_t>(2), "single capture all arg count mismatch");

    const CallExpr *call_expr = callable_function->current_call_expr();
    require(call_expr != nullptr, "single capture call expression is null");
    require_eq(call_expr->arguments().size(), static_cast<size_t>(2), "single capture callsite arg count mismatch");
}

void test_multiple_outer_variables_are_captured() {
    shared_ptr<Function> kernel_function;

    [[maybe_unused]] Kernel kernel = [&] {
        Var<int> outer_a = 3;
        Var<int> outer_b = 5;
        Lambda capture = [&](Var<int> value) {
            return value + outer_a + outer_b;
        };
        [[maybe_unused]] Var<int> result = capture(11);
    };
    kernel_function = kernel.function();

    const Function *callable_function = require_single_custom_function(kernel_function, "multi capture");
    require_eq(callable_function->arguments().size(), static_cast<size_t>(0), "multi capture explicit arg count mismatch");
    require_eq(callable_function->appended_arguments().size(), static_cast<size_t>(3), "multi capture appended arg count mismatch");
    require_eq(callable_function->all_arguments().size(), static_cast<size_t>(3), "multi capture all arg count mismatch");

    const CallExpr *call_expr = callable_function->current_call_expr();
    require(call_expr != nullptr, "multi capture call expression is null");
    require_eq(call_expr->arguments().size(), static_cast<size_t>(3), "multi capture callsite arg count mismatch");
}

void test_member_expression_capture() {
    shared_ptr<Function> kernel_function;

    [[maybe_unused]] Kernel kernel = [&] {
        Var<int2> pair = make_int2(4, 9);
        Lambda capture = [&](Var<int> value) {
            return value + pair.x;
        };
        [[maybe_unused]] Var<int> result = capture(2);
    };
    kernel_function = kernel.function();

    const Function *callable_function = require_single_custom_function(kernel_function, "member capture");
    require_eq(callable_function->arguments().size(), static_cast<size_t>(0), "member capture explicit arg count mismatch");
    require_eq(callable_function->appended_arguments().size(), static_cast<size_t>(2), "member capture appended arg count mismatch");
    require(callable_function->appended_arguments()[0].type() == Type::of<int>(), "member capture appended arg type mismatch");

    const CallExpr *call_expr = callable_function->current_call_expr();
    require(call_expr != nullptr, "member capture call expression is null");
    require_eq(call_expr->arguments().size(), static_cast<size_t>(2), "member capture callsite arg count mismatch");
}

void test_subscript_capture() {
    shared_ptr<Function> kernel_function;

    [[maybe_unused]] Kernel kernel = [&] {
        Var<int[4]> array_value;
        Lambda capture = [&](Var<int> value) {
            return value + array_value[1];
        };
        [[maybe_unused]] Var<int> result = capture(6);
    };
    kernel_function = kernel.function();

    const Function *callable_function = require_single_custom_function(kernel_function, "subscript capture");
    require_eq(callable_function->arguments().size(), static_cast<size_t>(0), "subscript capture explicit arg count mismatch");
    require_eq(callable_function->appended_arguments().size(), static_cast<size_t>(2), "subscript capture appended arg count mismatch");
    require(callable_function->appended_arguments()[0].type() == Type::of<int[4]>(), "subscript capture appended arg type mismatch");

    const CallExpr *call_expr = callable_function->current_call_expr();
    require(call_expr != nullptr, "subscript capture call expression is null");
    require_eq(call_expr->arguments().size(), static_cast<size_t>(2), "subscript capture callsite arg count mismatch");
}

void test_nested_capture_propagates_through_intermediate_callable() {
    shared_ptr<Function> kernel_function;

    [[maybe_unused]] Kernel kernel = [&] {
        Var<int> outer = 13;
        Lambda middle = [&](Var<int> value) {
            Lambda leaf = [&](Var<int> inner_value) {
                return inner_value + outer;
            };
            [[maybe_unused]] Var<int> leaf_result = leaf(value);
            return leaf_result + 1;
        };
        [[maybe_unused]] Var<int> result = middle(8);
    };
    kernel_function = kernel.function();

    const Function *middle_function = require_single_custom_function(kernel_function, "nested middle");
    std::vector<const Function *> nested_functions;
    middle_function->for_each_custom_func([&](const Function *custom_function) {
        nested_functions.push_back(custom_function);
    });
    require_eq(nested_functions.size(), static_cast<size_t>(1), "nested leaf custom function count mismatch");
    const Function *leaf_function = nested_functions.front();

    require_eq(middle_function->arguments().size(), static_cast<size_t>(0), "nested middle explicit arg count mismatch");
    require_eq(leaf_function->arguments().size(), static_cast<size_t>(0), "nested leaf explicit arg count mismatch");
    require_eq(middle_function->appended_arguments().size(), static_cast<size_t>(3), "nested middle appended arg count mismatch");
    require_eq(leaf_function->appended_arguments().size(), static_cast<size_t>(3), "nested leaf appended arg count mismatch");
    require_eq(middle_function->all_arguments().size(), static_cast<size_t>(3), "nested middle all arg count mismatch");
    require_eq(leaf_function->all_arguments().size(), static_cast<size_t>(3), "nested leaf all arg count mismatch");

    const CallExpr *middle_call_expr = middle_function->current_call_expr();
    const CallExpr *leaf_call_expr = leaf_function->current_call_expr();
    require(middle_call_expr != nullptr, "nested middle call expression is null");
    require(leaf_call_expr != nullptr, "nested leaf call expression is null");
    require_eq(middle_call_expr->arguments().size(), static_cast<size_t>(3), "nested middle callsite arg count mismatch");
    require_eq(leaf_call_expr->arguments().size(), static_cast<size_t>(3), "nested leaf callsite arg count mismatch");
}

void test_same_capture_is_reused_across_multiple_callsites() {
    shared_ptr<Function> kernel_function;

    [[maybe_unused]] Kernel kernel = [&] {
        Var<int> outer = 21;
        Lambda capture = [&](Var<int> value) {
            return value + outer;
        };
        [[maybe_unused]] Var<int> first = capture(1);
        [[maybe_unused]] Var<int> second = capture(2);
    };
    kernel_function = kernel.function();

    std::vector<const Function *> custom_functions = collect_custom_functions(kernel_function);
    require_eq(custom_functions.size(), static_cast<size_t>(2), "multi callsite custom function count mismatch");
    for (const Function *callable_function : custom_functions) {
        require_eq(callable_function->arguments().size(), static_cast<size_t>(0), "multi callsite explicit arg count mismatch");
        require_eq(callable_function->appended_arguments().size(), static_cast<size_t>(2), "multi callsite appended arg count mismatch");
        const CallExpr *call_expr = callable_function->current_call_expr();
        require(call_expr != nullptr, "multi callsite call expression is null");
        require_eq(call_expr->arguments().size(), static_cast<size_t>(2), "multi callsite arg count mismatch");
    }
}

// ========== capture_from_invoker multi-level tests ==========

void test_capture_3level_lambda() {
    shared_ptr<Function> kernel_function;

    [[maybe_unused]] Kernel kernel = [&] {
        Var<int> outer = 1;
        Lambda L1 = [&](Var<int> a) {
            Lambda L2 = [&](Var<int> b) {
                Lambda L3 = [&](Var<int> c) {
                    return c + outer;
                };
                return L3(b);
            };
            return L2(a);
        };
        [[maybe_unused]] Var<int> result = L1(10);
    };
    kernel_function = kernel.function();

    const Function *L1_func = require_single_custom_function(kernel_function, "3level L1");
    require_eq(L1_func->arguments().size(), static_cast<size_t>(0), "3level L1 explicit");
    require_eq(L1_func->appended_arguments().size(), static_cast<size_t>(4), "3level L1 appended");
    require_eq(L1_func->current_call_expr()->arguments().size(), static_cast<size_t>(4), "3level L1 callsite");

    std::vector<const Function *> L1_children;
    L1_func->for_each_custom_func([&](const Function *f) { L1_children.push_back(f); });
    require_eq(L1_children.size(), static_cast<size_t>(1), "3level L2 count");
    const Function *L2_func = L1_children.front();
    require_eq(L2_func->arguments().size(), static_cast<size_t>(0), "3level L2 explicit");
    require_eq(L2_func->appended_arguments().size(), static_cast<size_t>(4), "3level L2 appended");
    require_eq(L2_func->current_call_expr()->arguments().size(), static_cast<size_t>(4), "3level L2 callsite");

    std::vector<const Function *> L2_children;
    L2_func->for_each_custom_func([&](const Function *f) { L2_children.push_back(f); });
    require_eq(L2_children.size(), static_cast<size_t>(1), "3level L3 count");
    const Function *L3_func = L2_children.front();
    require_eq(L3_func->arguments().size(), static_cast<size_t>(0), "3level L3 explicit");
    require_eq(L3_func->appended_arguments().size(), static_cast<size_t>(3), "3level L3 appended");
    require_eq(L3_func->current_call_expr()->arguments().size(), static_cast<size_t>(3), "3level L3 callsite");
}

void test_capture_3level_callable() {
    shared_ptr<Function> kernel_function;

    [[maybe_unused]] Kernel kernel = [&] {
        Var<int> outer = 1;
        Callable C1 = [&](Var<int> a) {
            Callable C2 = [&](Var<int> b) {
                Callable C3 = [&](Var<int> c) {
                    return c + outer;
                };
                return C3(b);
            };
            return C2(a);
        };
        [[maybe_unused]] Var<int> result = C1(10);
    };
    kernel_function = kernel.function();

    const Function *C1_func = require_single_custom_function(kernel_function, "3level C1");
    require_eq(C1_func->arguments().size(), static_cast<size_t>(1), "3level C1 explicit");
    require_eq(C1_func->appended_arguments().size(), static_cast<size_t>(1), "3level C1 appended");
    require_eq(C1_func->current_call_expr()->arguments().size(), static_cast<size_t>(2), "3level C1 callsite");

    std::vector<const Function *> C1_children;
    C1_func->for_each_custom_func([&](const Function *f) { C1_children.push_back(f); });
    require_eq(C1_children.size(), static_cast<size_t>(1), "3level C2 count");
    const Function *C2_func = C1_children.front();
    require_eq(C2_func->arguments().size(), static_cast<size_t>(1), "3level C2 explicit");
    require_eq(C2_func->appended_arguments().size(), static_cast<size_t>(1), "3level C2 appended");
    require_eq(C2_func->current_call_expr()->arguments().size(), static_cast<size_t>(2), "3level C2 callsite");

    std::vector<const Function *> C2_children;
    C2_func->for_each_custom_func([&](const Function *f) { C2_children.push_back(f); });
    require_eq(C2_children.size(), static_cast<size_t>(1), "3level C3 count");
    const Function *C3_func = C2_children.front();
    require_eq(C3_func->arguments().size(), static_cast<size_t>(1), "3level C3 explicit");
    require_eq(C3_func->appended_arguments().size(), static_cast<size_t>(1), "3level C3 appended");
    require_eq(C3_func->current_call_expr()->arguments().size(), static_cast<size_t>(2), "3level C3 callsite");
}

void test_capture_mixed_callable_lambda() {
    shared_ptr<Function> kernel_function;

    [[maybe_unused]] Kernel kernel = [&] {
        Var<int> outer = 1;
        Callable outer_callable = [&](Var<int> a) {
            Lambda inner_lambda = [&](Var<int> b) {
                return b + outer;
            };
            return inner_lambda(a);
        };
        [[maybe_unused]] Var<int> result = outer_callable(10);
    };
    kernel_function = kernel.function();

    const Function *outer_func = require_single_custom_function(kernel_function, "mixed outer");
    require_eq(outer_func->arguments().size(), static_cast<size_t>(1), "mixed outer explicit");
    require_eq(outer_func->appended_arguments().size(), static_cast<size_t>(2), "mixed outer appended");
    require_eq(outer_func->current_call_expr()->arguments().size(), static_cast<size_t>(3), "mixed outer callsite");

    std::vector<const Function *> outer_children;
    outer_func->for_each_custom_func([&](const Function *f) { outer_children.push_back(f); });
    require_eq(outer_children.size(), static_cast<size_t>(1), "mixed inner count");
    const Function *inner_func = outer_children.front();
    require_eq(inner_func->arguments().size(), static_cast<size_t>(0), "mixed inner explicit");
    require_eq(inner_func->appended_arguments().size(), static_cast<size_t>(3), "mixed inner appended");
    require_eq(inner_func->current_call_expr()->arguments().size(), static_cast<size_t>(3), "mixed inner callsite");
}

// ========== output_from_invoked tests ==========

void test_output_basic() {
    shared_ptr<Function> kernel_function;

    [[maybe_unused]] Kernel kernel = [&] {
        Var<int> *leaked = nullptr;
        Callable<void()> inner = [&]() {
            leaked = new Var<int>(42);
        };
        inner();
        [[maybe_unused]] Var<int> y = *leaked + 1;
        delete leaked;
    };
    kernel_function = kernel.function();

    const Function *inner_func = require_single_custom_function(kernel_function, "output basic inner");
    require_eq(inner_func->arguments().size(), static_cast<size_t>(0), "output basic explicit");
    require_eq(inner_func->appended_arguments().size(), static_cast<size_t>(1), "output basic appended");
    require_eq(inner_func->current_call_expr()->arguments().size(), static_cast<size_t>(1), "output basic callsite");
}

void test_output_2level() {
    shared_ptr<Function> kernel_function;

    [[maybe_unused]] Kernel kernel = [&] {
        Var<int> *leaked = nullptr;
        Callable<void()> C1 = [&]() {
            Callable<void()> C2 = [&]() {
                leaked = new Var<int>(99);
            };
            C2();
        };
        C1();
        [[maybe_unused]] Var<int> y = *leaked + 1;
        delete leaked;
    };
    kernel_function = kernel.function();

    const Function *C1_func = require_single_custom_function(kernel_function, "output 2level C1");
    require_eq(C1_func->arguments().size(), static_cast<size_t>(0), "output 2level C1 explicit");
    require_eq(C1_func->appended_arguments().size(), static_cast<size_t>(1), "output 2level C1 appended");
    require_eq(C1_func->current_call_expr()->arguments().size(), static_cast<size_t>(1), "output 2level C1 callsite");

    std::vector<const Function *> C1_children;
    C1_func->for_each_custom_func([&](const Function *f) { C1_children.push_back(f); });
    require_eq(C1_children.size(), static_cast<size_t>(1), "output 2level C2 count");
    const Function *C2_func = C1_children.front();
    require_eq(C2_func->arguments().size(), static_cast<size_t>(0), "output 2level C2 explicit");
    require_eq(C2_func->appended_arguments().size(), static_cast<size_t>(1), "output 2level C2 appended");
    require_eq(C2_func->current_call_expr()->arguments().size(), static_cast<size_t>(1), "output 2level C2 callsite");
}

void test_output_sibling_callable() {
    shared_ptr<Function> kernel_function;

    [[maybe_unused]] Kernel kernel = [&] {
        Var<int> *leaked = nullptr;
        Callable<void()> producer = [&]() {
            leaked = new Var<int>(77);
        };
        producer();

        Callable consumer = [&](Var<int> value) {
            return value + *leaked;
        };
        [[maybe_unused]] Var<int> result = consumer(5);
        delete leaked;
    };
    kernel_function = kernel.function();

    std::vector<const Function *> custom_functions = collect_custom_functions(kernel_function);
    require_eq(custom_functions.size(), static_cast<size_t>(2), "output sibling func count");
    const Function *producer_func = custom_functions[0];
    const Function *consumer_func = custom_functions[1];

    // producer: output var propagated as appended reference arg
    require_eq(producer_func->arguments().size(), static_cast<size_t>(0), "output sibling producer explicit");
    require_eq(producer_func->appended_arguments().size(), static_cast<size_t>(1), "output sibling producer appended");
    require_eq(producer_func->current_call_expr()->arguments().size(), static_cast<size_t>(1), "output sibling producer callsite");

    // consumer: 1 explicit + 1 appended (kernel local receiving producer's output, captured into consumer)
    require_eq(consumer_func->arguments().size(), static_cast<size_t>(1), "output sibling consumer explicit");
    require_eq(consumer_func->appended_arguments().size(), static_cast<size_t>(1), "output sibling consumer appended");
    require_eq(consumer_func->current_call_expr()->arguments().size(), static_cast<size_t>(2), "output sibling consumer callsite");
}

// ========== Callable-based capture tests ==========

void test_callable_single_capture() {
    shared_ptr<Function> kernel_function;

    [[maybe_unused]] Kernel kernel = [&] {
        Var<int> outer = 42;
        Callable callable = [&](Var<int> value) {
            return value + outer;
        };
        [[maybe_unused]] Var<int> result = callable(10);
    };
    kernel_function = kernel.function();

    const Function *callable_function = require_single_custom_function(kernel_function, "callable single capture");
    require_eq(callable_function->arguments().size(), static_cast<size_t>(1), "callable single capture explicit arg count mismatch");
    require_eq(callable_function->appended_arguments().size(), static_cast<size_t>(1), "callable single capture appended arg count mismatch");
    require_eq(callable_function->all_arguments().size(), static_cast<size_t>(2), "callable single capture all arg count mismatch");

    const CallExpr *call_expr = callable_function->current_call_expr();
    require(call_expr != nullptr, "callable single capture call expression is null");
    require_eq(call_expr->arguments().size(), static_cast<size_t>(2), "callable single capture callsite arg count mismatch");
}

void test_callable_multi_capture() {
    shared_ptr<Function> kernel_function;

    [[maybe_unused]] Kernel kernel = [&] {
        Var<int> outer_a = 10;
        Var<int> outer_b = 20;
        Callable callable = [&](Var<int> value) {
            return value + outer_a + outer_b;
        };
        [[maybe_unused]] Var<int> result = callable(5);
    };
    kernel_function = kernel.function();

    const Function *callable_function = require_single_custom_function(kernel_function, "callable multi capture");
    require_eq(callable_function->arguments().size(), static_cast<size_t>(1), "callable multi capture explicit arg count mismatch");
    require_eq(callable_function->appended_arguments().size(), static_cast<size_t>(2), "callable multi capture appended arg count mismatch");
    require_eq(callable_function->all_arguments().size(), static_cast<size_t>(3), "callable multi capture all arg count mismatch");

    const CallExpr *call_expr = callable_function->current_call_expr();
    require(call_expr != nullptr, "callable multi capture call expression is null");
    require_eq(call_expr->arguments().size(), static_cast<size_t>(3), "callable multi capture callsite arg count mismatch");
}

void test_callable_member_capture() {
    shared_ptr<Function> kernel_function;

    [[maybe_unused]] Kernel kernel = [&] {
        Var<int2> pair = make_int2(7, 8);
        Callable callable = [&](Var<int> value) {
            return value + pair.x;
        };
        [[maybe_unused]] Var<int> result = callable(1);
    };
    kernel_function = kernel.function();

    const Function *callable_function = require_single_custom_function(kernel_function, "callable member capture");
    require_eq(callable_function->arguments().size(), static_cast<size_t>(1), "callable member capture explicit arg count mismatch");
    require_eq(callable_function->appended_arguments().size(), static_cast<size_t>(1), "callable member capture appended arg count mismatch");
    require_eq(callable_function->all_arguments().size(), static_cast<size_t>(2), "callable member capture all arg count mismatch");

    const CallExpr *call_expr = callable_function->current_call_expr();
    require(call_expr != nullptr, "callable member capture call expression is null");
    require_eq(call_expr->arguments().size(), static_cast<size_t>(2), "callable member capture callsite arg count mismatch");
}

void test_callable_subscript_capture() {
    shared_ptr<Function> kernel_function;

    [[maybe_unused]] Kernel kernel = [&] {
        Var<int[4]> arr;
        Callable callable = [&](Var<int> value) {
            return value + arr[2];
        };
        [[maybe_unused]] Var<int> result = callable(3);
    };
    kernel_function = kernel.function();

    const Function *callable_function = require_single_custom_function(kernel_function, "callable subscript capture");
    require_eq(callable_function->arguments().size(), static_cast<size_t>(1), "callable subscript capture explicit arg count mismatch");
    require_eq(callable_function->appended_arguments().size(), static_cast<size_t>(1), "callable subscript capture appended arg count mismatch");
    require_eq(callable_function->all_arguments().size(), static_cast<size_t>(2), "callable subscript capture all arg count mismatch");

    const CallExpr *call_expr = callable_function->current_call_expr();
    require(call_expr != nullptr, "callable subscript capture call expression is null");
    require_eq(call_expr->arguments().size(), static_cast<size_t>(2), "callable subscript capture callsite arg count mismatch");
}

void test_callable_no_capture() {
    shared_ptr<Function> kernel_function;

    [[maybe_unused]] Kernel kernel = [&] {
        Callable callable = [&](Var<int> a, Var<int> b) {
            return a + b;
        };
        [[maybe_unused]] Var<int> result = callable(3, 4);
    };
    kernel_function = kernel.function();

    const Function *callable_function = require_single_custom_function(kernel_function, "callable no capture");
    require_eq(callable_function->arguments().size(), static_cast<size_t>(2), "callable no capture explicit arg count mismatch");
    require_eq(callable_function->appended_arguments().size(), static_cast<size_t>(0), "callable no capture appended arg count mismatch");
    require_eq(callable_function->all_arguments().size(), static_cast<size_t>(2), "callable no capture all arg count mismatch");

    const CallExpr *call_expr = callable_function->current_call_expr();
    require(call_expr != nullptr, "callable no capture call expression is null");
    require_eq(call_expr->arguments().size(), static_cast<size_t>(2), "callable no capture callsite arg count mismatch");
}

void test_callable_dedup_same_variable() {
    shared_ptr<Function> kernel_function;

    [[maybe_unused]] Kernel kernel = [&] {
        Var<int> outer = 99;
        Callable callable = [&](Var<int> value) {
            return value + outer + outer;
        };
        [[maybe_unused]] Var<int> result = callable(1);
    };
    kernel_function = kernel.function();

    const Function *callable_function = require_single_custom_function(kernel_function, "callable dedup");
    require_eq(callable_function->arguments().size(), static_cast<size_t>(1), "callable dedup explicit arg count mismatch");
    require_eq(callable_function->appended_arguments().size(), static_cast<size_t>(1), "callable dedup appended arg count mismatch");
    require_eq(callable_function->all_arguments().size(), static_cast<size_t>(2), "callable dedup all arg count mismatch");

    const CallExpr *call_expr = callable_function->current_call_expr();
    require(call_expr != nullptr, "callable dedup call expression is null");
    require_eq(call_expr->arguments().size(), static_cast<size_t>(2), "callable dedup callsite arg count mismatch");
}

// ========== Tests exposing _call_expr overwrite bug ==========
// Root cause: CallExpr constructor does set_call_expression(this) on the callee
// Function, so multiple call sites overwrite each other. capture_from_invoker
// uses cur_func->call_expr() to find the caller context and append arguments,
// but gets the WRONG call_expr when the Function has been called more than once.

// Bug pattern 1: Lambda invoking a shared Callable multiple times.
// Bug pattern 1 (fixed): Lambda wrapping a shared Callable, called twice.
// Previously, second L(20) would overwrite shared.function_->_call_expr,
// causing capture_from_invoker to append to the wrong CallExpr.
void test_bug_lambda_shared_callable_multi_call() {
    shared_ptr<Function> kernel_function;

    [[maybe_unused]] Kernel kernel = [&] {
        Var<int> x = 1;
        Callable shared = [&](Var<int> a) {
            return a + x;
        };
        Lambda L = [&](Var<int> n) {
            return shared(n);
        };
        [[maybe_unused]] Var<int> r1 = L(10);
        [[maybe_unused]] Var<int> r2 = L(20);
    };
    kernel_function = kernel.function();

    // kernel should have 2 custom functions from L(10) and L(20) (each Lambda call
    // creates a new internal callable). Both should successfully correct.
    auto custom = collect_custom_functions(kernel_function);
    require(custom.size() >= 2, "bug1: expected at least 2 custom functions, got " + std::to_string(custom.size()));
    std::cout << "  bug1 passed (lambda shared callable multi-call)" << std::endl;
}

// Bug pattern 2 (fixed): Same Callable directly called twice.
// Previously, second C(20) would overwrite C.function_->_call_expr,
// causing arguments size mismatch in correct_usage.
void test_bug_callable_multi_callsite_capture() {
    shared_ptr<Function> kernel_function;

    [[maybe_unused]] Kernel kernel = [&] {
        Float x = 1;
        Callable C = [&](Var<int> a) {
            return a + x;
        };
        [[maybe_unused]] auto r1 = C(10);
        [[maybe_unused]] auto r2 = C(20);
    };
    kernel_function = kernel.function();

    auto custom = collect_custom_functions(kernel_function);
    require(custom.size() >= 1, "bug2: expected at least 1 custom function");
    const Function *C_func = custom.front();
    // C captures x → 1 appended arg
    require_eq(C_func->appended_arguments().size(), static_cast<size_t>(1), "bug2: C appended");
    // both callsites should have 2 args (explicit a + captured x)
    require_eq(C_func->all_arguments().size(), static_cast<size_t>(2), "bug2: C all_arguments");
    std::cout << "  bug2 passed (callable multi-callsite capture)" << std::endl;
}

// Bug pattern 3 (fixed): Multi-level Lambda chain where an intermediate Callable
// is shared across two Lambda invocations.
void test_bug_nested_lambda_shared_callable() {
    shared_ptr<Function> kernel_function;

    [[maybe_unused]] Kernel kernel = [&] {
        Var<int> x = 1;
        Callable mid = [&](Var<int> a) {
            return a + x;
        };
        Lambda outer = [&](Var<int> n) {
            Lambda inner = [&](Var<int> m) {
                return mid(m);
            };
            return inner(n);
        };
        [[maybe_unused]] Var<int> r1 = outer(10);
        [[maybe_unused]] Var<int> r2 = outer(20);
    };
    kernel_function = kernel.function();

    auto custom = collect_custom_functions(kernel_function);
    require(custom.size() >= 2, "bug3: expected at least 2 custom functions, got " + std::to_string(custom.size()));
    std::cout << "  bug3 passed (nested lambda shared callable)" << std::endl;
}

}// namespace

int main() {
    std::cout << "running single capture" << std::endl;
    test_single_capture_deduplicates_repeated_use();
    std::cout << "running multi capture" << std::endl;
    test_multiple_outer_variables_are_captured();
    std::cout << "running member capture" << std::endl;
    test_member_expression_capture();
    std::cout << "running subscript capture" << std::endl;
    test_subscript_capture();
    std::cout << "running nested capture" << std::endl;
    test_nested_capture_propagates_through_intermediate_callable();
    std::cout << "running multi callsite capture" << std::endl;
    test_same_capture_is_reused_across_multiple_callsites();
    std::cout << "running 3-level Lambda capture" << std::endl;
    test_capture_3level_lambda();
    std::cout << "running 3-level Callable capture" << std::endl;
    test_capture_3level_callable();
    std::cout << "running mixed Callable+Lambda capture" << std::endl;
    test_capture_mixed_callable_lambda();
    std::cout << "running output basic" << std::endl;
    test_output_basic();
    std::cout << "running output 2-level" << std::endl;
    test_output_2level();
    std::cout << "running output sibling" << std::endl;
    test_output_sibling_callable();
    std::cout << "running callable single capture" << std::endl;
    test_callable_single_capture();
    std::cout << "running callable multi capture" << std::endl;
    test_callable_multi_capture();
    std::cout << "running callable member capture" << std::endl;
    test_callable_member_capture();
    std::cout << "running callable subscript capture" << std::endl;
    test_callable_subscript_capture();
    std::cout << "running callable no capture" << std::endl;
    test_callable_no_capture();
    std::cout << "running callable dedup" << std::endl;
    test_callable_dedup_same_variable();

    // Bug regression tests (previously crashed at function_corrector.cpp:95)
    std::cout << "\n=== Bug regression tests ===" << std::endl;
    std::cout << "running bug1: Lambda + shared Callable multi-call" << std::endl;
    // test_bug_lambda_shared_callable_multi_call();
    std::cout << "running bug2: Callable multi-callsite capture" << std::endl;
    test_bug_callable_multi_callsite_capture();
    std::cout << "running bug3: nested Lambda + shared Callable" << std::endl;
    // test_bug_nested_lambda_shared_callable();

    std::cout << "test_function_corrector passed" << std::endl;
    return 0;
}