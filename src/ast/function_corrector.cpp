//
// Created by Zero on 2023/12/3.
//

#include "function_corrector.h"

namespace ocarina {

void FunctionCorrector::traverse(Function &function) noexcept {
    visit(function.body());
    function.correct_used_structures();
    if (function.is_callable()) {
        bool valid = function.check_context();
        OC_ERROR_IF_NOT(valid, "FunctionCorrector error: invalid function ", function.description().c_str());
    }
}

CallExpr *FunctionCorrector::find_call_expr_for(const Function *func) const noexcept {
    for (int i = function_stack_.size() - 1; i >= 0; --i) {
        if (function_stack_.at(i) == func) {
            return call_expr_stack_.at(i - 1);
        }
    }
    OC_ERROR("Cannot find call expr for function");
    return nullptr;
}


void FunctionCorrector::apply(Function *function, int counter) noexcept {
    function_stack_.push_back(function);
    traverse(*current_function());
    if (current_function()->is_kernel()) {
        /// Split parameter structure into separate elements
        stage_ = SplitParamStruct;
        current_function()->splitting_arguments();
        traverse(*current_function());
        stage_ = ProcessCapture;
    }
    if (function->is_kernel()) {
        bool valid = function->check_context();
        OC_ERROR_IF_NOT(valid, "FunctionCorrector error: invalid function ", function->description().c_str());
    }
    function_stack_.pop_back();
}

bool FunctionCorrector::is_from_invoker(const Expression *expression) noexcept {
    return std::find(function_stack_.begin(), function_stack_.end(),
                     expression->context()) != function_stack_.end();
}

void FunctionCorrector::process_capture(const Expression *&expression, Function *cur_func) noexcept {
    if (expression->context() == cur_func) {
        return;
    }
    auto bit_or = [](Usage lhs, Usage rhs) {
        return Usage(to_underlying(lhs) | to_underlying(rhs));
    };
    const Expression *old_expr = expression;
    if (is_from_invoker(expression)) {
        capture_from_invoker(expression, cur_func);
    } else {
        output_from_invoked(expression, cur_func);
    }
}

void FunctionCorrector::visit_expr(const Expression *const &expression, Function *cur_func) noexcept {
    cur_func = cur_func == nullptr ? current_function() : cur_func;
    if (expression == nullptr) {
        return;
    }

    switch (expression->tag()) {
        case Expression::Tag::REF: {
            static_cast<const VariableExpr *>(expression)->variable().mark_used();
            process_capture(const_cast<const Expression *&>(expression), cur_func);
            break;
        }
        case Expression::Tag::MEMBER: {
            static_cast<const VariableExpr *>(expression)->variable().mark_used();
            process_member_expr(const_cast<const Expression *&>(expression), cur_func);
            break;
        }
        default: {
            expression->accept(*this);
            break;
        }
    }
}

void FunctionCorrector::capture_from_invoker(const Expression *&expression, Function *cur_func) noexcept {
    bool contain;
    const Expression *old_expr = expression;
    expression = cur_func->mapping_captured_argument(expression, &contain);
    if (contain) {
        return;
    }
    captured_outer_exprs_[cur_func].push_back(old_expr);
    CallExpr *call_expr = find_call_expr_for(cur_func);
    visit_expr(old_expr, const_cast<Function *>(call_expr->context()));
    call_expr->append_argument(old_expr);
}

namespace detail {

void correct_usage(const CallExpr *expr) noexcept {
    vector<Variable> formal_arguments = expr->function()->all_arguments();
    OC_ERROR_IF(expr->arguments().size() != formal_arguments.size());

    auto bit_or = [](Usage lhs, Usage rhs) {
        return Usage(to_underlying(lhs) | to_underlying(rhs));
    };

    for (int i = 0; i < formal_arguments.size(); ++i) {
        Variable formal_arg = formal_arguments[i];
        Usage &formal_arg_usage = const_cast<Usage &>(expr->function()->variable_usage(formal_arg.uid()));

        const Expression *act_arg = expr->argument(i);
        Usage act_arg_usage = act_arg->usage();

        Usage combined = bit_or(formal_arg_usage, act_arg_usage);
        if (act_arg->type()->is_resource()) {
            formal_arg_usage = combined;
            act_arg->mark(combined);
        } else {
            act_arg->mark(combined);
        }
    }
}

}// namespace detail

void FunctionCorrector::visit(const CallExpr *const_expr) {
    CallExpr *expr = const_cast<CallExpr *>(const_expr);
    for (const Expression *const &arg : expr->arguments_) {
        visit_expr(arg);
    }
    if (!expr->function_) {
        return;
    }
    call_expr_stack_.push_back(expr);
    apply(const_cast<Function *>(expr->function_));
    call_expr_stack_.pop_back();
    // When a function was already corrected from a previous call site,
    // its body is patched and traverse won't re-discover captures.
    // Propagate the missing captured args to this CallExpr.
    const Function *func = expr->function_;
    size_t expected = func->all_arguments().size();
    if (expr->arguments().size() < expected) {
        auto it = captured_outer_exprs_.find(func);
        if (it != captured_outer_exprs_.end()) {
            for (const Expression *original_outer : it->second) {
                const Expression *propagated = original_outer;
                visit_expr(propagated);
                expr->append_argument(propagated);
            }
        }
    }
    detail::correct_usage(expr);
}

void FunctionCorrector::output_from_invoked(const Expression *&expression, Function *cur_func) noexcept {
    auto context = const_cast<Function *>(expression->context());
    Function *invoked = context;
    Function *invoker = nullptr;
    bool in_path = false;
    context->append_output_argument(expression, nullptr);
    const Expression *org_expr = expression;

    /// foreach invoke path,output target variable layer by layer to the kernel (top-level function)
    while (true) {
        CallExpr *call_expr = const_cast<CallExpr *>(invoked->current_call_expr());
        invoker = const_cast<Function *>(call_expr->context());
        if (invoked == cur_func || invoker == cur_func) {
            in_path = true;
        }
        bool contain;
        const RefExpr *ref_expr;
        if (invoker == kernel()) {
            ref_expr = invoker->mapping_local_variable(org_expr, &contain);
            if (!contain) {
                call_expr->append_argument(ref_expr);
            }
            break;
        } else {
            ref_expr = invoker->mapping_output_argument(org_expr, &contain);
            if (!contain) {
                call_expr->append_argument(ref_expr);
            }
        }
        invoked = invoker;
    }
    if (in_path) {
        if (cur_func == kernel()) {
            expression = cur_func->outer_to_local(org_expr);
        } else {
            expression = cur_func->outer_to_argument(org_expr);
        }
    } else {
        const RefExpr *kernel_expr = kernel()->outer_to_local(expression);
        expression = kernel_expr;
        capture_from_invoker(expression, cur_func);
    }
}

void FunctionCorrector::visit(const ScopeStmt *scope) {
    for (const Statement *stmt : scope->statements()) {
        stmt->accept(*this);
    }
}

void FunctionCorrector::visit(const AssignStmt *stmt) {
    visit_expr(stmt->lhs_);
    OC_ERROR_IF(stmt->lhs()->type()->is_resource());
    stmt->lhs()->mark(Usage::WRITE);
    visit_expr(stmt->rhs_);
}

void FunctionCorrector::visit(const ExprStmt *stmt) {
    visit_expr(stmt->expression_);
}

void FunctionCorrector::visit(const ForStmt *stmt) {
    visit_expr(stmt->var_);
    visit_expr(stmt->condition_);
    visit_expr(stmt->step_);
    stmt->body()->accept(*this);
}

void FunctionCorrector::visit(const IfStmt *stmt) {
    visit_expr(stmt->condition_);
    stmt->true_branch()->accept(*this);
    stmt->false_branch()->accept(*this);
}

void FunctionCorrector::visit(const LoopStmt *stmt) {
    stmt->body()->accept(*this);
}

void FunctionCorrector::visit(const ReturnStmt *stmt) {
    visit_expr(stmt->expression_);
}

void FunctionCorrector::visit(const SwitchCaseStmt *stmt) {
    visit_expr(stmt->expr_);
    stmt->body()->accept(*this);
}

void FunctionCorrector::visit(const SwitchStmt *stmt) {
    visit_expr(stmt->expression_);
    stmt->body()->accept(*this);
}

void FunctionCorrector::visit(const SwitchDefaultStmt *stmt) {
    stmt->body()->accept(*this);
}

void FunctionCorrector::visit(const BinaryExpr *expr) {
    visit_expr(expr->lhs_);
    visit_expr(expr->rhs_);
}

void FunctionCorrector::visit(const CastExpr *expr) {
    visit_expr(expr->expression_);
}

void FunctionCorrector::visit(const ConditionalExpr *expr) {
    visit_expr(expr->pred_);
    visit_expr(expr->true__);
    visit_expr(expr->false__);
}

void FunctionCorrector::visit(const MemberExpr *expr) {
    OC_ERROR_IF(stage_ == ProcessCapture);
}

void FunctionCorrector::process_member_expr(const Expression *&expression, Function *cur_func) noexcept {
    auto member_expr = dynamic_cast<const MemberExpr *>(expression);
    switch (stage_) {
        case ProcessCapture:
            if (member_expr->context() == cur_func) {
                visit_expr(member_expr->parent_, cur_func);
            } else {
                process_capture(expression, cur_func);
            }
            break;
        case SplitParamStruct:
            if (member_expr->parent()->type()->is_param_struct()) {
                process_param_struct(expression);
            } else {
                visit_expr(member_expr->parent_);
            }
            break;
        default:
            break;
    }
}

void FunctionCorrector::process_param_struct(const Expression *&expression) noexcept {
    const MemberExpr *member_expr = dynamic_cast<const MemberExpr *>(expression);
    const Expression *parent = member_expr->parent();
    vector<int> path;
    path.push_back(member_expr->member_index());
    do {
        const MemberExpr *member_parent = dynamic_cast<const MemberExpr *>(parent);
        if (member_parent) {
            parent = member_parent->parent();
            path.push_back(member_parent->member_index());
        }
    } while (parent->is_member());
    const RefExpr *ref_expr = dynamic_cast<const RefExpr *>(parent);
    path.push_back(ref_expr->variable().uid());
    std::reverse(path.begin(), path.end());
    kernel()->replace_param_struct_member(path, expression);
}

void FunctionCorrector::visit(const SubscriptExpr *expr) {
    for (const Expression *const &index : expr->indexes_) {
        visit_expr(index);
    }
    visit_expr(expr->range_);
}

void FunctionCorrector::visit(const UnaryExpr *expr) {
    visit_expr(expr->operand_);
}

}// namespace ocarina