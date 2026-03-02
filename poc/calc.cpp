#include <cctype>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <format>
#include <functional>
#include <generator>
#include <iterator>
#include <memory>
#include <numbers>
#include <print>
#include <ranges>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

template <class... Ts> struct overloaded : Ts... { using Ts::operator()...; };

struct Token {
    enum class Kind {
        Ident, Number,
        Plus, Minus,
        Star, Slash,          // NEW
        Caret,
        SuperscriptTwo, SuperscriptThree,
        LParen, RParen,
        End, Unknown
    } kind;
    std::string_view lexeme;
    size_t pos;
};

constexpr std::string_view to_string(Token::Kind k) {
    using enum Token::Kind;
    switch (k) {
        case Ident:   return "Ident";
        case Number:  return "Number";
        case Plus:    return "Plus";
        case Minus:   return "Minus";
        case Star:    return "Star";
        case Slash:   return "Slash";
        case Caret:   return "Caret";
        case SuperscriptTwo: return "SuperscriptTwo";
        case SuperscriptThree: return "SuperscriptThree";
        case LParen:  return "LParen";
        case RParen:  return "RParen";
        case End:     return "End";
        case Unknown: return "Unknown";
    }
    std::unreachable();
}

static bool is_ident_start(unsigned char c) { return std::isalpha(c) || c == '_'; }
static bool is_ident_cont (unsigned char c) { return std::isalnum(c) || c == '_'; }

std::generator<Token> lex(std::string_view s) {
    using enum Token::Kind;
    size_t i = 0;

    auto peek = [&](size_t k = 0) -> char {
        return (i + k < s.size()) ? s[i + k] : '\0';
    };

    auto take_while = [&](auto pred) {
        size_t start = i;
        while (i < s.size() && pred((unsigned char)s[i])) ++i;
        return s.substr(start, i - start);
    };

    while (i < s.size()) {
        if (std::isspace((unsigned char)peek())) { ++i; continue; }

        size_t start = i;
        char c = peek();

        if (is_ident_start((unsigned char)c)) {
            auto lexeme = take_while(is_ident_cont);
            co_yield {Ident, lexeme, start};
            continue;
        }

        if (std::isdigit((unsigned char)c)) {
            auto lexeme = take_while([](unsigned char ch){ return std::isdigit(ch); });

            if (peek() == '.' && std::isdigit((unsigned char)peek(1))) {
                ++i; // consume '.'
                (void)take_while([](unsigned char ch){ return std::isdigit(ch); });
                lexeme = s.substr(start, i - start);
            }

            co_yield {Number, lexeme, start};
            continue;
        }

        // UTF-8 superscripts ² (U+00B2) and ³ (U+00B3)
        if (i + 1 < s.size()) {
            unsigned char b0 = (unsigned char)s[i];
            unsigned char b1 = (unsigned char)s[i + 1];

            if (b0 == 0xC2 && b1 == 0xB2) { // ²
                auto lexeme = s.substr(start, 2);
                i += 2;
                co_yield {SuperscriptTwo, lexeme, start};
                continue;
            }
            if (b0 == 0xC2 && b1 == 0xB3) { // ³
                auto lexeme = s.substr(start, 2);
                i += 2;
                co_yield {SuperscriptThree, lexeme, start};
                continue;
            }
        }

        ++i;
        auto lexeme = s.substr(start, 1);
        switch (c) {
            case '+': co_yield {Plus,  lexeme, start}; break;
            case '-': co_yield {Minus, lexeme, start}; break;
            case '*': co_yield {Star,  lexeme, start}; break; // NEW
            case '/': co_yield {Slash, lexeme, start}; break; // NEW
            case '^': co_yield {Caret, lexeme, start}; break;
            case '(': co_yield {LParen,lexeme, start}; break;
            case ')': co_yield {RParen,lexeme, start}; break;
            default:  co_yield {Unknown,lexeme, start}; break;
        }
    }

    co_yield {End, {}, s.size()};
}

struct Expr {
    struct Binary {
        enum class Op { Add, Sub, Mul, Div, Pow } op;
        std::unique_ptr<Expr> lhs, rhs;
    };
    struct Unary  { Token op; std::unique_ptr<Expr> rhs; };
    struct Number { long double value; };
    struct Variable { std::string name; };

    std::variant<Binary, Unary, Number, Variable> node;

    static Expr number(long double v) { return {Number{v}}; }
    static Expr variable(std::string name) { return {Variable{std::move(name)}}; }
    static Expr binary(Binary::Op op, Expr a, Expr b) {
        return {
            Binary{op,
                std::make_unique<Expr>(std::move(a)),
                std::make_unique<Expr>(std::move(b))
            }
        };
    }
    static Expr unary(Token op, Expr rhs) {
        return { Unary{op, std::make_unique<Expr>(std::move(rhs)) } };
    }

    template <class... Fs>
    decltype(auto) visit(Fs&&... fs) {
        return std::visit(overloaded{std::forward<Fs>(fs)...}, node);
    }

    template <class... Fs>
    decltype(auto) visit(Fs&&... fs) const {
        return std::visit(overloaded{std::forward<Fs>(fs)...}, node);
    }
};

constexpr std::string_view to_string(Expr::Binary::Op op) {
    using enum Expr::Binary::Op;
    switch (op) {
        case Add: return "+";
        case Sub: return "-";
        case Mul: return "*";
        case Div: return "/";
        case Pow: return "^";
    }
    std::unreachable();
}

struct ParseError : std::runtime_error {
    size_t pos;
    ParseError(size_t p, std::string msg)
        : std::runtime_error(std::move(msg)), pos(p) {}
};

class Parser {
    std::generator<Token> gen_;
    decltype(gen_.begin()) it_;
    Token tok_{};

    Token consume() { Token t = tok_; ++it_; tok_ = *it_; return t; }

    [[noreturn]] void error_here(std::string msg) const {
        throw ParseError(tok_.pos, std::move(msg));
    }

    void expect(Token::Kind k, std::string msg) {
        if (tok_.kind != k) error_here(std::move(msg));
        consume();
    }

    using enum Token::Kind;

    Expr expression() { return term(); }

    Expr term() {
        Expr lhs = factor();
        while (tok_.kind == Plus || tok_.kind == Minus) {
            Token op = consume();
            lhs = Expr::binary(
                op.kind == Plus ? Expr::Binary::Op::Add : Expr::Binary::Op::Sub,
                std::move(lhs),
                factor()
            );
        }
        return lhs;
    }

    Expr factor() {                     // * and / (higher precedence)  // NEW
        Expr lhs = power();
        while (true) {
            if (tok_.kind == Star || tok_.kind == Slash) {
                Token op = consume();
                lhs = Expr::binary(
                    op.kind == Star ? Expr::Binary::Op::Mul : Expr::Binary::Op::Div,
                    std::move(lhs),
                    power()
                );
                continue;
            }

            if (tok_.kind == Ident && std::holds_alternative<Expr::Number>(lhs.node)) {
                lhs = Expr::binary(
                    Expr::Binary::Op::Mul,
                    std::move(lhs),
                    power()
                );
                continue;
            }

            break;
        }
        return lhs;
    }

    Expr power() {
        Expr lhs = postfix();
        if (tok_.kind == Caret) {
            consume();
            lhs = Expr::binary(Expr::Binary::Op::Pow, std::move(lhs), power()); // right-associative
        }
        return lhs;
    }

    Expr postfix() { // NEW
        Expr lhs = unary();
        while (tok_.kind == SuperscriptTwo || tok_.kind == SuperscriptThree) {
            Token t = consume();
            long double exp = (t.kind == SuperscriptTwo) ? 2.0L : 3.0L;
            lhs = Expr::binary(Expr::Binary::Op::Pow, std::move(lhs), Expr::number(exp));
        }
        return lhs;
    }

    Expr unary() {
        if (tok_.kind == Minus) {
            Token op = consume();
            return Expr::unary(op, unary());
        }
        return primary();
    }

    Expr primary() {
        switch (tok_.kind) {
            case Number: {
                Token t = consume();
                long double v{};
                auto first = t.lexeme.data();
                auto last  = first + t.lexeme.size();
                auto [ptr, ec] = std::from_chars(first, last, v);
                if (ec != std::errc{} || ptr != last)
                    error_here(std::format("invalid number: '{}'", t.lexeme));
                return Expr::number(v);
            }
            case Ident: {
                Token t = consume();
                return Expr::variable(std::string(t.lexeme));
            }
            case LParen: {
                consume();
                Expr e = expression();
                expect(RParen, "expected ')'");
                return e;
            }
            case Unknown:
                error_here(std::format("unknown token: '{}'", tok_.lexeme));
            case End:
                error_here("unexpected end of input");
            default:
                error_here(std::format("unexpected token: '{}'", tok_.lexeme));
        }
    }

public:
    explicit Parser(std::string_view input)
        : gen_(lex(input)), it_(gen_.begin()) {
        tok_ = *it_;
    }

    [[nodiscard]] Expr parse() {
        Expr e = expression();
        expect(End, "expected end of input");
        return e;
    }
};

Expr parse(std::string_view s) {
    Parser parser(s);
    return parser.parse();
}

struct DAG {
    struct Op {
        std::string op;
        std::vector<size_t> args;
    };

    std::vector<long double> constants;
    std::vector<Op> ops;
};

/// Lowers an expression AST into a topologically-sorted DAG “program” (a linear IR).
/// Each AST node becomes an Op; Call nodes reference their operands by indices into
/// the returned vector. The order is postorder/topological: operands always appear
/// before their uses, and the last element computes the overall (outermost) expression.
DAG lower_expr_to_toposorted_dag(const Expr& expr) {
    struct Builder {
        DAG dag;

        size_t build_node(const Expr& e) { return std::visit(*this, e.node); }

        size_t operator()(const Expr::Number& num) {
            size_t cidx = dag.constants.size();
            if (auto iter = std::ranges::find(dag.constants, num.value); iter != dag.constants.end()) {
                cidx = std::distance(dag.constants.begin(), iter);
            } else {
                dag.constants.emplace_back(num.value);
            } 
            size_t idx = dag.ops.size();
            dag.ops.emplace_back(DAG::Op{"Constant", {cidx}});
            return idx;
        }

        size_t operator()(const Expr::Variable&) {
            size_t idx = dag.ops.size();
            dag.ops.emplace_back(DAG::Op{"Control", {}});
            return idx;
        }

        size_t operator()(const Expr::Unary& un) {
            if (un.op.kind != Token::Kind::Minus)
                throw std::runtime_error(std::format("unsupported unary operator: '{}'", un.op.lexeme));

            size_t rhs = build_node(*un.rhs);

            size_t idx = dag.ops.size();
            dag.ops.emplace_back(DAG::Op{"neg", { rhs } });
            return idx;
        }

        size_t operator()(const Expr::Binary& bin) {
            size_t lhs = build_node(*bin.lhs);
            size_t rhs = build_node(*bin.rhs);

            size_t idx = dag.ops.size();
            dag.ops.emplace_back(DAG::Op{std::string{to_string(bin.op)}, { lhs, rhs } });
            return idx;
        }
    };

    Builder b;
    (void)b.build_node(expr);
    return std::move(b.dag);
}

long double eval(const Expr& expr, const std::unordered_map<std::string, long double> env) {
    return expr.visit(
        [](const Expr::Number& num) -> long double {
            return num.value;
        },
        [&](const Expr::Variable& var) -> long double {
            auto it = env.find(var.name);
            if (it == env.end())
                throw std::runtime_error(std::format("undefined variable: '{}'", var.name));
            return it->second;
        },
        [&](const Expr::Unary& unary) -> long double {
            if (unary.op.kind == Token::Kind::Minus) return -eval(*unary.rhs, env);
            throw std::runtime_error(std::format("unsupported unary operator: '{}'", unary.op.lexeme));
        },
        [&](const Expr::Binary& binary) -> long double {
            auto a = eval(*binary.lhs, env);
            auto b = eval(*binary.rhs, env);
            using enum Expr::Binary::Op;
            switch (binary.op) {
                case Add: return a + b;
                case Sub: return a - b;
                case Mul: return a * b;
                case Div: return a / b;
                case Pow: return std::powl(a, b);
            }
            std::unreachable();
        }
    );
}

void print(const Expr& expr, size_t indent = 0) {
    for (size_t i = 0; i < indent; ++i) std::print("  ");

    expr.visit(
        [](const Expr::Number& num) {
            std::println("Number(value={})", num.value);
        },
        [](const Expr::Variable& var) {
            std::println("Variable(name='{}')", var.name);
        },
        [indent](const Expr::Unary& unary) {
            std::println("Unary(op='{}')", unary.op.lexeme);
            print(*unary.rhs, indent + 1);
        },
        [indent](const Expr::Binary& binary) {
            std::println("Binary(op='{}')", to_string(binary.op));
            print(*binary.lhs, indent + 1);
            print(*binary.rhs, indent + 1);
        }
    );
}

void print(DAG const& dag) {
    std::println("DAG {{");

    std::println("  constants ({}):", dag.constants.size());
    for (size_t i = 0; i < dag.constants.size(); ++i) {
        std::println("    [{}] {}", i, dag.constants[i]);
    }

    std::println("  ops ({}):", dag.ops.size());
    for (size_t i = 0; i < dag.ops.size(); ++i) {
        auto const& op = dag.ops[i];
        std::print("    [{}] {}(", i, op.op);
        for (size_t j = 0; j < op.args.size(); ++j) {
            if (j) std::print(", ");
            std::print("{}", op.args[j]);
        }
        std::println(")");
    }

    std::println("}}");
}

int main(int argc, const char *argv[]) {
    if (argc != 2) return 1;
    try {
        auto expr = parse(argv[1]);
        print(expr);
        print(lower_expr_to_toposorted_dag(expr));


        const std::unordered_map<std::string, long double> env = {
            {"e", std::numbers::e_v<long double>},
            {"log2e", std::numbers::log2e_v<long double>},
            {"log10e", std::numbers::log10e_v<long double>},
            {"pi", std::numbers::pi_v<long double>},
            {"tau", std::numbers::pi_v<long double> * 2},
            {"inv_pi", std::numbers::inv_pi_v<long double>},
            {"inv_sqrtpi", std::numbers::inv_sqrtpi_v<long double>},
            {"ln2", std::numbers::ln2_v<long double>},
            {"ln10", std::numbers::ln10_v<long double>},
            {"sqrt2", std::numbers::sqrt2_v<long double>},
            {"sqrt3", std::numbers::sqrt3_v<long double>},
            {"inv_sqrt3", std::numbers::inv_sqrt3_v<long double>},
            {"egamma", std::numbers::egamma_v<long double>},
            {"phi", std::numbers::phi_v<long double>},
        };

        std::println("-> {}", eval(expr, env));
    } catch (const ParseError& e) {
        std::println("Parse error at pos {}: {}", e.pos, e.what());
    } catch (const std::exception& e) {
        std::println("Exception: {}", e.what());
    }
}
