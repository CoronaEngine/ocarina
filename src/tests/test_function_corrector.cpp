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
    const CallExpr *call_expr = function->call_expr();
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

    const CallExpr *call_expr = callable_function->call_expr();
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

    const CallExpr *call_expr = callable_function->call_expr();
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

    const CallExpr *call_expr = callable_function->call_expr();
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

    const CallExpr *call_expr = callable_function->call_expr();
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

    const CallExpr *middle_call_expr = middle_function->call_expr();
    const CallExpr *leaf_call_expr = leaf_function->call_expr();
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
        const CallExpr *call_expr = callable_function->call_expr();
        require(call_expr != nullptr, "multi callsite call expression is null");
        require_eq(call_expr->arguments().size(), static_cast<size_t>(2), "multi callsite arg count mismatch");
    }
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

    const CallExpr *call_expr = callable_function->call_expr();
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

    const CallExpr *call_expr = callable_function->call_expr();
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

    const CallExpr *call_expr = callable_function->call_expr();
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

    const CallExpr *call_expr = callable_function->call_expr();
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

    const CallExpr *call_expr = callable_function->call_expr();
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

    const CallExpr *call_expr = callable_function->call_expr();
    require(call_expr != nullptr, "callable dedup call expression is null");
    require_eq(call_expr->arguments().size(), static_cast<size_t>(2), "callable dedup callsite arg count mismatch");
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
    std::cout << "test_function_corrector passed" << std::endl;
    return 0;
}