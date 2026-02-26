#include <libgccjit.h>
#include <assert.h>
#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const size_t BS_V = 128;

static void die(gcc_jit_context *ctx, const char *msg)
{
    fprintf(stderr, "%s\n", msg);
    fprintf(stderr, "%s\n", gcc_jit_context_get_first_error(ctx));
    exit(1);
}

static gcc_jit_rvalue *get_address_of_first_array_element(gcc_jit_lvalue *lvalue)
{
    gcc_jit_context *ctx = gcc_jit_object_get_context(gcc_jit_lvalue_as_object(lvalue));
    gcc_jit_type *t_size_t = gcc_jit_context_get_type(ctx, GCC_JIT_TYPE_SIZE_T);
    gcc_jit_rvalue *rv_zero = gcc_jit_context_new_rvalue_from_long(ctx, t_size_t, 0);
    gcc_jit_lvalue *access = gcc_jit_context_new_array_access(ctx, NULL,
        gcc_jit_lvalue_as_rvalue(lvalue), rv_zero
    );
    return gcc_jit_lvalue_get_address(access, NULL);
}

/* --- Graph descriptor --- */

struct vertex
{
    const char *name;
    char rate;
    size_t nArgs;
    const size_t *args;
};

struct dag
{
    const float *constants;
    const struct vertex *vertices;
    const size_t nVertices;
};

static const float CONSTS[] = { 0.0f, 0.2f };

static const size_t V0_ARGS[] = { 0 };
static const size_t V1_ARGS[] = { 0 };
static const size_t V2_ARGS[] = { 0, 1 };
static const size_t V3_ARGS[] = { 1 };
static const size_t V4_ARGS[] = { 2, 3 };

static const struct vertex VERTICES[] = {
  { "Control", 'b', 1, V0_ARGS },
  { "Const", 'b', 1, V1_ARGS },
  { "SinOsc", 'a', 2, V2_ARGS },
  { "Const", 'b', 1, V3_ARGS },
  { "Mul", 'a', 2, V4_ARGS }
};

static struct dag graph = {
  .constants = CONSTS,
  .vertices = VERTICES,
  .nVertices = 5
};

/* --- Opcode registry --- */

typedef struct Opcode Opcode;

typedef void (*opcode_init_fn)(const Opcode *op, gcc_jit_context *ctx, gcc_jit_type *t_state, unsigned int sample_rate);
typedef gcc_jit_type *(*make_state_type_fn)(const Opcode *op, gcc_jit_context *ctx);
typedef void (*emit_init_fn)(const Opcode *op, gcc_jit_context *ctx, gcc_jit_block *entry, gcc_jit_lvalue *lv_state_field);

struct Arg {
    gcc_jit_rvalue *rv; /* scalar float or float* depending on rate */
    char rate;
};

typedef gcc_jit_rvalue *(*emit_proc_fn)(
    const Opcode *op,
    const struct dag *graph,
    size_t vertex_index,
    gcc_jit_context *ctx,
    gcc_jit_function *fn_process,
    gcc_jit_block *entry,
    gcc_jit_lvalue *lv_state_field,
    gcc_jit_rvalue *rv_controls,
    struct Arg *args, size_t n_args,
    char out_rate
);

struct Opcode {
    const char *name;
    void *priv;                         /* private per-opcode data */
    int special_indices;                /* if true, vertex args are indices (not dependency vertex indices) */
    opcode_init_fn init_fn;             /* optional (may be NULL); called before anything else */
    make_state_type_fn make_state_type; /* optional (may be NULL) */
    emit_init_fn emit_init;             /* optional (may be NULL) */
    emit_proc_fn emit_proc;
};

static Opcode *g_opcodes = NULL;
static size_t g_n_opcodes = 0;

static int opcode_cmp_by_name(const void *a, const void *b)
{
    const Opcode *oa = (const Opcode *)a;
    const Opcode *ob = (const Opcode *)b;
    return strcmp(oa->name, ob->name);
}

static char *mangle(const struct vertex *v, size_t i)
{
    const struct vertex *cur = &v[i];

    size_t name_len = strlen(cur->name);

    size_t n_in = cur->nArgs;
    if (strcmp(cur->name, "Const") == 0 || strcmp(cur->name, "Control") == 0)
        n_in = 0;

    /* "<name>" + "_" + <n_in chars> + <1 out char> + "\0" */
    size_t len = name_len + 1 + n_in + 1 + 1;

    char *s = (char *)malloc(len);
    if (!s) return NULL;

    memcpy(s, cur->name, name_len);
    s[name_len] = '_';

    for (size_t k = 0; k < n_in; k++) {
        size_t arg_i = cur->args[k];
        s[name_len + 1 + k] = v[arg_i].rate;
    }

    s[name_len + 1 + n_in] = cur->rate;
    s[name_len + 1 + n_in + 1] = '\0';

    return s;
}

static const Opcode *find_opcode(const struct vertex *v, size_t i)
{
    if (!g_opcodes || g_n_opcodes == 0) {
        fprintf(stderr, "opcode registry is empty\n");
        exit(1);
    }

    char *key_name = mangle(v, i);
    if (!key_name) {
        fprintf(stderr, "failed to mangle opcode name for vertex %zu\n", i);
        exit(1);
    }

    Opcode key = { .name = key_name };
    const Opcode *res = bsearch(&key, g_opcodes, g_n_opcodes, sizeof(Opcode), opcode_cmp_by_name);

    if (!res) {
        fprintf(stderr, "opcode not found: %s\n", key_name);
        free(key_name);
        exit(1);
    }

    free(key_name);
    return res;
}

static int register_opcode(Opcode op)
{
    Opcode *new_ops = realloc(g_opcodes, (g_n_opcodes + 1) * sizeof(*new_ops));
    if (!new_ops) return 0;
    g_opcodes = new_ops;
    g_opcodes[g_n_opcodes++] = op;
    qsort(g_opcodes, g_n_opcodes, sizeof(Opcode), opcode_cmp_by_name);
    return 1;
}

/* --- Const --- */

static gcc_jit_rvalue *const_emit_proc(
    const Opcode *,
    const struct dag *graph,
    size_t vertex_index,
    gcc_jit_context *ctx,
    gcc_jit_function *,
    gcc_jit_block *,
    gcc_jit_lvalue *,
    gcc_jit_rvalue *,
    struct Arg *, size_t n_args,
    char out_rate
)
{
    assert(out_rate == 'b');
    assert(n_args == 0);
    const struct vertex *v = &graph->vertices[vertex_index];
    assert(v->nArgs == 1);

    return gcc_jit_context_new_rvalue_from_double(ctx,
        gcc_jit_context_get_type(ctx, GCC_JIT_TYPE_FLOAT),
        graph->constants[v->args[0]]
    );
}

/* --- Control --- */

static gcc_jit_rvalue *control_emit_proc(
    const Opcode *,
    const struct dag *graph,
    size_t vertex_index,
    gcc_jit_context *ctx,
    gcc_jit_function *,
    gcc_jit_block *,
    gcc_jit_lvalue *,
    gcc_jit_rvalue *rv_controls,
    struct Arg *, size_t n_args,
    char out_rate
)
{
    assert(out_rate == 'b');
    assert(n_args == 0);
    const struct vertex *v = &graph->vertices[vertex_index];
    assert(v->nArgs == 1);

    gcc_jit_type *t_size_t = gcc_jit_context_get_type(ctx, GCC_JIT_TYPE_SIZE_T);
    gcc_jit_rvalue *idx =
        gcc_jit_context_new_rvalue_from_long(ctx, t_size_t, (long)v->args[0]);

    gcc_jit_lvalue *lv =
        gcc_jit_context_new_array_access(ctx, NULL, rv_controls, idx);

    return gcc_jit_lvalue_as_rvalue(lv);
}

/* --- SinOsc: has per-instance state { float phase; } --- */

struct sinosc_priv {
    gcc_jit_field *fld_phase;
    gcc_jit_struct *st_sinosc;
    gcc_jit_function *kernel_bba;
    gcc_jit_function *kernel_aba;
};

static gcc_jit_type *sinosc_make_state_type(const Opcode *op, gcc_jit_context *ctx)
{
    struct sinosc_priv *p = op->priv;

    if (!p->fld_phase) {
        gcc_jit_type *t_float = gcc_jit_context_get_type(ctx, GCC_JIT_TYPE_FLOAT);
        p->fld_phase = gcc_jit_context_new_field(ctx, NULL, t_float, "phase");
    }
    if (!p->st_sinosc) {
        p->st_sinosc = gcc_jit_context_new_struct_type(ctx, NULL, "sinosc_state", 1, &p->fld_phase);
    }
    return gcc_jit_struct_as_type(p->st_sinosc);
}

static void sinosc_emit_init(const Opcode *op, gcc_jit_context *ctx, gcc_jit_block *entry, gcc_jit_lvalue *lv_state)
{
    struct sinosc_priv *p = op->priv;
    gcc_jit_type *t_float = gcc_jit_context_get_type(ctx, GCC_JIT_TYPE_FLOAT);

    assert(p->fld_phase);

    gcc_jit_lvalue *lv_phase =
        gcc_jit_lvalue_access_field(lv_state, NULL, p->fld_phase);

    gcc_jit_rvalue *c_0_f = gcc_jit_context_new_rvalue_from_double(ctx, t_float, 0.0);
    gcc_jit_block_add_assignment(entry, NULL, lv_phase, c_0_f);
}

static void sinosc_init(const Opcode *op, gcc_jit_context *ctx, gcc_jit_type *t_state, unsigned int sample_rate)
{
    struct sinosc_priv *p = (struct sinosc_priv *)op->priv;

    const int want_bba = (strcmp(op->name, "SinOsc_bba") == 0);
    const int want_aba = (strcmp(op->name, "SinOsc_aba") == 0);

    if (!want_bba && !want_aba) {
        fprintf(stderr, "sinosc_init called for unexpected opcode: %s\n", op->name);
        exit(1);
    }

    if ((want_bba && p->kernel_bba) || (want_aba && p->kernel_aba))
        return;

    gcc_jit_type *t_void   = gcc_jit_context_get_type(ctx, GCC_JIT_TYPE_VOID);
    gcc_jit_type *t_size_t = gcc_jit_context_get_type(ctx, GCC_JIT_TYPE_SIZE_T);
    gcc_jit_type *t_float  = gcc_jit_context_get_type(ctx, GCC_JIT_TYPE_FLOAT);

    gcc_jit_type *t_float_ptr = gcc_jit_type_get_pointer(t_float);
    gcc_jit_type *t_const_float_ptr = gcc_jit_type_get_pointer(gcc_jit_type_get_const(t_float));
    gcc_jit_type *t_state_ptr = gcc_jit_type_get_pointer(t_state);

    gcc_jit_rvalue *c_0_size  = gcc_jit_context_new_rvalue_from_long(ctx, t_size_t, 0);
    gcc_jit_rvalue *c_BS_size = gcc_jit_context_new_rvalue_from_long(ctx, t_size_t, (long)BS_V);

    gcc_jit_rvalue *c_TAU = gcc_jit_context_new_rvalue_from_double(ctx,
        t_float, 2.0 * M_PI
    );
    gcc_jit_rvalue *c_SR =
        gcc_jit_context_new_rvalue_from_double(ctx, t_float, (double)sample_rate);

    assert(p->fld_phase);

    /* External: float sinf(float) */
    gcc_jit_param *p_sinf_x = gcc_jit_context_new_param(ctx, NULL, t_float, "x");
    gcc_jit_function *fn_sinf =
        gcc_jit_context_new_function(ctx, NULL,
                                     GCC_JIT_FUNCTION_IMPORTED,
                                     t_float, "sinf",
                                     1, &p_sinf_x, 0);

    gcc_jit_param *p_ps_state =
        gcc_jit_context_new_param(ctx, NULL, t_state_ptr, "state");
    gcc_jit_param *p_ps_output =
        gcc_jit_context_new_param(ctx, NULL, t_float_ptr, "output");

    gcc_jit_param *p_ps_freq =
        gcc_jit_context_new_param(ctx, NULL, want_aba ? t_const_float_ptr : t_float, "frequency");
    gcc_jit_param *p_ps_off =
        gcc_jit_context_new_param(ctx, NULL, t_float, "phase_offset");

    gcc_jit_param *ps_params[] = { p_ps_state, p_ps_output, p_ps_freq, p_ps_off };

    gcc_jit_function **out_kernel = want_bba ? &p->kernel_bba : &p->kernel_aba;
    const char *kernel_name = want_bba ? "sinosc_process_bba" : "sinosc_process_aba";

    *out_kernel =
        gcc_jit_context_new_function(ctx, NULL, GCC_JIT_FUNCTION_INTERNAL,
                                     t_void, kernel_name,
                                     4, ps_params, 0);

    gcc_jit_block *entry     = gcc_jit_function_new_block(*out_kernel, "entry");
    gcc_jit_block *loop_cond = gcc_jit_function_new_block(*out_kernel, "loop_cond");
    gcc_jit_block *loop_body = gcc_jit_function_new_block(*out_kernel, "loop_body");
    gcc_jit_block *loop_inc  = gcc_jit_function_new_block(*out_kernel, "loop_inc");
    gcc_jit_block *done      = gcc_jit_function_new_block(*out_kernel, "done");

    gcc_jit_lvalue *lv_phase =
        gcc_jit_function_new_local(*out_kernel, NULL, t_float, "phase");
    gcc_jit_lvalue *lv_inc =
        gcc_jit_function_new_local(*out_kernel, NULL, t_float, "phase_increment");
    gcc_jit_lvalue *lv_i =
        gcc_jit_function_new_local(*out_kernel, NULL, t_size_t, "i");

    gcc_jit_lvalue *lv_state_phase =
        gcc_jit_lvalue_access_field(
            gcc_jit_rvalue_dereference(gcc_jit_param_as_rvalue(p_ps_state), NULL),
            NULL, p->fld_phase);

    gcc_jit_block_add_assignment(entry, NULL, lv_phase, gcc_jit_lvalue_as_rvalue(lv_state_phase));

    if (want_bba) {
        gcc_jit_rvalue *num =
            gcc_jit_context_new_binary_op(ctx, NULL, GCC_JIT_BINARY_OP_MULT, t_float,
                                          c_TAU, gcc_jit_param_as_rvalue(p_ps_freq));
        gcc_jit_rvalue *inc =
            gcc_jit_context_new_binary_op(ctx, NULL, GCC_JIT_BINARY_OP_DIVIDE, t_float,
                                          num, c_SR);
        gcc_jit_block_add_assignment(entry, NULL, lv_inc, inc);
    }

    gcc_jit_block_add_assignment(entry, NULL, lv_i, c_0_size);
    gcc_jit_block_end_with_jump(entry, NULL, loop_cond);

    gcc_jit_rvalue *cond =
        gcc_jit_context_new_comparison(ctx, NULL, GCC_JIT_COMPARISON_LT,
                                       gcc_jit_lvalue_as_rvalue(lv_i),
                                       c_BS_size);
    gcc_jit_block_end_with_conditional(loop_cond, NULL, cond, loop_body, done);

    if (want_aba) {
        gcc_jit_lvalue *lv_freq_i =
            gcc_jit_context_new_array_access(ctx, NULL,
                                             gcc_jit_param_as_rvalue(p_ps_freq),
                                             gcc_jit_lvalue_as_rvalue(lv_i));

        gcc_jit_rvalue *num_i =
            gcc_jit_context_new_binary_op(ctx, NULL, GCC_JIT_BINARY_OP_MULT, t_float,
                                          c_TAU, gcc_jit_lvalue_as_rvalue(lv_freq_i));
        gcc_jit_rvalue *inc_i =
            gcc_jit_context_new_binary_op(ctx, NULL, GCC_JIT_BINARY_OP_DIVIDE, t_float,
                                          num_i, c_SR);
        gcc_jit_block_add_assignment(loop_body, NULL, lv_inc, inc_i);
    }

    /* out[i] = sinf(phase + phase_offset); */
    gcc_jit_rvalue *phase_plus_off =
        gcc_jit_context_new_binary_op(ctx, NULL, GCC_JIT_BINARY_OP_PLUS, t_float,
                                      gcc_jit_lvalue_as_rvalue(lv_phase),
                                      gcc_jit_param_as_rvalue(p_ps_off));

    gcc_jit_rvalue *call_sinf =
        gcc_jit_context_new_call(ctx, NULL, fn_sinf, 1, &phase_plus_off);

    gcc_jit_lvalue *lv_out_i =
        gcc_jit_context_new_array_access(ctx, NULL,
                                         gcc_jit_param_as_rvalue(p_ps_output),
                                         gcc_jit_lvalue_as_rvalue(lv_i));
    gcc_jit_block_add_assignment(loop_body, NULL, lv_out_i, call_sinf);

    gcc_jit_rvalue *phase_next =
        gcc_jit_context_new_binary_op(ctx, NULL, GCC_JIT_BINARY_OP_PLUS, t_float,
                                      gcc_jit_lvalue_as_rvalue(lv_phase),
                                      gcc_jit_lvalue_as_rvalue(lv_inc));
    gcc_jit_block_add_assignment(loop_body, NULL, lv_phase, phase_next);

    gcc_jit_block_end_with_jump(loop_body, NULL, loop_inc);

    gcc_jit_rvalue *i_next =
        gcc_jit_context_new_binary_op(ctx, NULL, GCC_JIT_BINARY_OP_PLUS, t_size_t,
                                      gcc_jit_lvalue_as_rvalue(lv_i),
                                      gcc_jit_context_new_rvalue_from_long(ctx, t_size_t, 1));
    gcc_jit_block_add_assignment(loop_inc, NULL, lv_i, i_next);
    gcc_jit_block_end_with_jump(loop_inc, NULL, loop_cond);

    gcc_jit_block_add_assignment(done, NULL, lv_state_phase, gcc_jit_lvalue_as_rvalue(lv_phase));
    gcc_jit_block_end_with_void_return(done, NULL);
}

static gcc_jit_rvalue *sinosc_emit_proc(
    const Opcode *op,
    const struct dag *graph,
    size_t vertex_index,
    gcc_jit_context *ctx,
    gcc_jit_function *fn_process,
    gcc_jit_block *entry,
    gcc_jit_lvalue *lv_state,
    gcc_jit_rvalue *rv_controls,
    struct Arg *args, size_t n_args,
    char out_rate
)
{
    (void)graph;
    (void)rv_controls;

    assert(out_rate == 'a');
    assert(n_args == 2);

    struct sinosc_priv *p = op->priv;

    const int want_bba = (strcmp(op->name, "SinOsc_bba") == 0);
    const int want_aba = (strcmp(op->name, "SinOsc_aba") == 0);
    assert(want_bba || want_aba);

    assert((want_bba && p && p->kernel_bba) || (want_aba && p && p->kernel_aba));

    assert(args[0].rate == (want_aba ? 'a' : 'b'));
    assert(args[1].rate == 'b');

    gcc_jit_function *kernel = want_bba ? p->kernel_bba : p->kernel_aba;

    gcc_jit_type *t_float = gcc_jit_context_get_type(ctx, GCC_JIT_TYPE_FLOAT);
    gcc_jit_type *t_float_array_BS =
        gcc_jit_context_new_array_type(ctx, NULL, t_float, (int)BS_V);

    char name[32];
    snprintf(name, sizeof(name), "e%zu", vertex_index);
    gcc_jit_lvalue *lv_buf =
        gcc_jit_function_new_local(fn_process, NULL, t_float_array_BS, name);

    gcc_jit_rvalue *rv_state_ptr = gcc_jit_lvalue_get_address(lv_state, NULL);

    gcc_jit_rvalue *call_args[] = {
        rv_state_ptr,
        get_address_of_first_array_element(lv_buf),
        args[0].rv,
        args[1].rv
    };

    gcc_jit_block_add_eval(entry, NULL,
        gcc_jit_context_new_call(ctx, NULL, kernel, 4, call_args));

    return get_address_of_first_array_element(lv_buf);
}

/* --- BinOp: supports signatures: _aaa, _aba, _baa, _bbb --- */

struct binop_priv {
    gcc_jit_function *kernel;
    enum gcc_jit_binary_op op_kind;
    char sig[4]; /* "aaa", "aba", "baa", "bbb" */
};

static enum gcc_jit_binary_op binop_kind_from_name(const char *opcode_name)
{
    if (strncmp(opcode_name, "Mul_", 4) == 0) return GCC_JIT_BINARY_OP_MULT;
    if (strncmp(opcode_name, "Add_", 4) == 0) return GCC_JIT_BINARY_OP_PLUS;
    if (strncmp(opcode_name, "Sub_", 4) == 0) return GCC_JIT_BINARY_OP_MINUS;
    if (strncmp(opcode_name, "Div_", 4) == 0) return GCC_JIT_BINARY_OP_DIVIDE;
    fprintf(stderr, "unknown binop opcode: %s\n", opcode_name);
    exit(1);
}

static void binop_sig_from_name(const char *opcode_name, char out_sig[4])
{
    const char *u = strrchr(opcode_name, '_');
    if (!u || strlen(u + 1) != 3) {
        fprintf(stderr, "bad binop opcode signature: %s\n", opcode_name);
        exit(1);
    }
    out_sig[0] = u[1];
    out_sig[1] = u[2];
    out_sig[2] = u[3];
    out_sig[3] = '\0';

    if (strcmp(out_sig, "aaa") && strcmp(out_sig, "aba") &&
        strcmp(out_sig, "baa") && strcmp(out_sig, "bbb")) {
        fprintf(stderr, "unsupported binop signature: %s\n", out_sig);
        exit(1);
    }
}

static void binop_init(const Opcode *op, gcc_jit_context *ctx, gcc_jit_type *t_state, unsigned int sample_rate)
{
    (void)t_state;
    (void)sample_rate;

    struct binop_priv *p = (struct binop_priv *)op->priv;
    if (p->kernel)
        return;

    p->op_kind = binop_kind_from_name(op->name);
    binop_sig_from_name(op->name, p->sig);

    gcc_jit_type *t_void   = gcc_jit_context_get_type(ctx, GCC_JIT_TYPE_VOID);
    gcc_jit_type *t_size_t = gcc_jit_context_get_type(ctx, GCC_JIT_TYPE_SIZE_T);
    gcc_jit_type *t_float  = gcc_jit_context_get_type(ctx, GCC_JIT_TYPE_FLOAT);

    gcc_jit_type *t_float_ptr       = gcc_jit_type_get_pointer(t_float);
    gcc_jit_type *t_const_float_ptr = gcc_jit_type_get_pointer(gcc_jit_type_get_const(t_float));

    gcc_jit_rvalue *c_0_size  = gcc_jit_context_new_rvalue_from_long(ctx, t_size_t, 0);
    gcc_jit_rvalue *c_BS_size = gcc_jit_context_new_rvalue_from_long(ctx, t_size_t, (long)BS_V);

    char fn_name[64];
    snprintf(fn_name, sizeof(fn_name), "%s_process_%s", op->name, p->sig);

    /* bbb: handled inline in binop_emit_proc (no kernel) */
    if (strcmp(p->sig, "bbb") == 0) {
        p->kernel = NULL;
        return;
    }

    /* otherwise: void f(float *r, <a>, <b>) where <a>/<b> are float or float* depending */
    gcc_jit_param *p_r = gcc_jit_context_new_param(ctx, NULL, t_float_ptr, "r");

    gcc_jit_type *t_a = (p->sig[0] == 'a') ? t_const_float_ptr : t_float;
    gcc_jit_type *t_b = (p->sig[1] == 'a') ? t_const_float_ptr : t_float;

    gcc_jit_param *p_a = gcc_jit_context_new_param(ctx, NULL, t_a, "a");
    gcc_jit_param *p_b = gcc_jit_context_new_param(ctx, NULL, t_b, "b");
    gcc_jit_param *params[] = { p_r, p_a, p_b };

    p->kernel =
        gcc_jit_context_new_function(ctx, NULL, GCC_JIT_FUNCTION_INTERNAL,
                                     t_void, fn_name, 3, params, 0);

    gcc_jit_block *entry = gcc_jit_function_new_block(p->kernel, "entry");
    gcc_jit_block *cond  = gcc_jit_function_new_block(p->kernel, "cond");
    gcc_jit_block *body  = gcc_jit_function_new_block(p->kernel, "body");
    gcc_jit_block *inc   = gcc_jit_function_new_block(p->kernel, "inc");
    gcc_jit_block *done  = gcc_jit_function_new_block(p->kernel, "done");

    gcc_jit_lvalue *lv_i = gcc_jit_function_new_local(p->kernel, NULL, t_size_t, "i");
    gcc_jit_block_add_assignment(entry, NULL, lv_i, c_0_size);
    gcc_jit_block_end_with_jump(entry, NULL, cond);

    gcc_jit_rvalue *cnd =
        gcc_jit_context_new_comparison(ctx, NULL, GCC_JIT_COMPARISON_LT,
                                       gcc_jit_lvalue_as_rvalue(lv_i), c_BS_size);
    gcc_jit_block_end_with_conditional(cond, NULL, cnd, body, done);

    gcc_jit_lvalue *lv_r_i =
        gcc_jit_context_new_array_access(ctx, NULL,
                                         gcc_jit_param_as_rvalue(p_r),
                                         gcc_jit_lvalue_as_rvalue(lv_i));

    gcc_jit_rvalue *ra =
        (p->sig[0] == 'a')
        ? gcc_jit_lvalue_as_rvalue(
              gcc_jit_context_new_array_access(ctx, NULL,
                                               gcc_jit_param_as_rvalue(p_a),
                                               gcc_jit_lvalue_as_rvalue(lv_i)))
        : gcc_jit_param_as_rvalue(p_a);

    gcc_jit_rvalue *rb =
        (p->sig[1] == 'a')
        ? gcc_jit_lvalue_as_rvalue(
              gcc_jit_context_new_array_access(ctx, NULL,
                                               gcc_jit_param_as_rvalue(p_b),
                                               gcc_jit_lvalue_as_rvalue(lv_i)))
        : gcc_jit_param_as_rvalue(p_b);

    gcc_jit_rvalue *expr =
        gcc_jit_context_new_binary_op(ctx, NULL, p->op_kind, t_float, ra, rb);

    gcc_jit_block_add_assignment(body, NULL, lv_r_i, expr);
    gcc_jit_block_end_with_jump(body, NULL, inc);

    gcc_jit_rvalue *i_next =
        gcc_jit_context_new_binary_op(ctx, NULL, GCC_JIT_BINARY_OP_PLUS, t_size_t,
                                      gcc_jit_lvalue_as_rvalue(lv_i),
                                      gcc_jit_context_new_rvalue_from_long(ctx, t_size_t, 1));
    gcc_jit_block_add_assignment(inc, NULL, lv_i, i_next);
    gcc_jit_block_end_with_jump(inc, NULL, cond);

    gcc_jit_block_end_with_void_return(done, NULL);
}

static gcc_jit_rvalue *binop_emit_proc(
    const Opcode *op,
    const struct dag *graph,
    size_t vertex_index,
    gcc_jit_context *ctx,
    gcc_jit_function *fn_process,
    gcc_jit_block *entry,
    gcc_jit_lvalue *lv_state,
    gcc_jit_rvalue *rv_controls,
    struct Arg *args, size_t n_args,
    char out_rate
)
{
    (void)graph;
    (void)vertex_index;
    (void)lv_state;
    (void)rv_controls;

    assert(n_args == 2);

    struct binop_priv *p = op->priv;
    assert(p);

    gcc_jit_type *t_float = gcc_jit_context_get_type(ctx, GCC_JIT_TYPE_FLOAT);

    if (strcmp(p->sig, "bbb") == 0) {
        assert(out_rate == 'b');
        assert(args[0].rate == 'b');
        assert(args[1].rate == 'b');

        char name[32];
        snprintf(name, sizeof(name), "e%zu", vertex_index);
        gcc_jit_lvalue *lv_tmp =
            gcc_jit_function_new_local(fn_process, NULL, t_float, name);

        gcc_jit_rvalue *expr =
            gcc_jit_context_new_binary_op(ctx, NULL, p->op_kind, t_float,
                                          args[0].rv, args[1].rv);

        gcc_jit_block_add_assignment(entry, NULL, lv_tmp, expr);
        return gcc_jit_lvalue_as_rvalue(lv_tmp);
    }

    assert(p->kernel);
    assert(out_rate == 'a');

    gcc_jit_type *t_float_array_BS =
        gcc_jit_context_new_array_type(ctx, NULL, t_float, (int)BS_V);

    char name[32];
    snprintf(name, sizeof(name), "e%zu", vertex_index);
    gcc_jit_lvalue *lv_buf =
        gcc_jit_function_new_local(fn_process, NULL, t_float_array_BS, name);

    gcc_jit_rvalue *call_args[] = {
        get_address_of_first_array_element(lv_buf),
        args[0].rv,
        args[1].rv
    };

    gcc_jit_block_add_eval(entry, NULL,
        gcc_jit_context_new_call(ctx, NULL, p->kernel, 3, call_args));

    return get_address_of_first_array_element(lv_buf);
}

/* Call once before compile */
static void register_builtin_opcodes(void)
{
    static struct sinosc_priv sinosc_p = {0};

    static struct binop_priv mul_aaa = {0};
    static struct binop_priv mul_aba = {0};
    static struct binop_priv mul_baa = {0};
    static struct binop_priv mul_bbb = {0};

    static struct binop_priv add_aaa = {0};
    static struct binop_priv add_aba = {0};
    static struct binop_priv add_baa = {0};
    static struct binop_priv add_bbb = {0};

    static struct binop_priv sub_aaa = {0};
    static struct binop_priv sub_aba = {0};
    static struct binop_priv sub_baa = {0};
    static struct binop_priv sub_bbb = {0};

    static struct binop_priv div_aaa = {0};
    static struct binop_priv div_aba = {0};
    static struct binop_priv div_baa = {0};
    static struct binop_priv div_bbb = {0};

    register_opcode((Opcode){
        .name = "SinOsc_bba",
        .priv = &sinosc_p,
        .special_indices = 0,
        .init_fn = sinosc_init,
        .make_state_type = sinosc_make_state_type,
        .emit_init = sinosc_emit_init,
        .emit_proc = sinosc_emit_proc
    });
    register_opcode((Opcode){
        .name = "SinOsc_aba",
        .priv = &sinosc_p,
        .special_indices = 0,
        .init_fn = sinosc_init,
        .make_state_type = sinosc_make_state_type,
        .emit_init = sinosc_emit_init,
        .emit_proc = sinosc_emit_proc
    });

    register_opcode((Opcode){ .name = "Mul_aaa", .priv = &mul_aaa, .special_indices = 0, .init_fn = binop_init, .make_state_type = NULL, .emit_init = NULL, .emit_proc = binop_emit_proc });
    register_opcode((Opcode){ .name = "Mul_aba", .priv = &mul_aba, .special_indices = 0, .init_fn = binop_init, .make_state_type = NULL, .emit_init = NULL, .emit_proc = binop_emit_proc });
    register_opcode((Opcode){ .name = "Mul_baa", .priv = &mul_baa, .special_indices = 0, .init_fn = binop_init, .make_state_type = NULL, .emit_init = NULL, .emit_proc = binop_emit_proc });
    register_opcode((Opcode){ .name = "Mul_bbb", .priv = &mul_bbb, .special_indices = 0, .init_fn = binop_init, .make_state_type = NULL, .emit_init = NULL, .emit_proc = binop_emit_proc });

    register_opcode((Opcode){ .name = "Add_aaa", .priv = &add_aaa, .special_indices = 0, .init_fn = binop_init, .make_state_type = NULL, .emit_init = NULL, .emit_proc = binop_emit_proc });
    register_opcode((Opcode){ .name = "Add_aba", .priv = &add_aba, .special_indices = 0, .init_fn = binop_init, .make_state_type = NULL, .emit_init = NULL, .emit_proc = binop_emit_proc });
    register_opcode((Opcode){ .name = "Add_baa", .priv = &add_baa, .special_indices = 0, .init_fn = binop_init, .make_state_type = NULL, .emit_init = NULL, .emit_proc = binop_emit_proc });
    register_opcode((Opcode){ .name = "Add_bbb", .priv = &add_bbb, .special_indices = 0, .init_fn = binop_init, .make_state_type = NULL, .emit_init = NULL, .emit_proc = binop_emit_proc });

    register_opcode((Opcode){ .name = "Sub_aaa", .priv = &sub_aaa, .special_indices = 0, .init_fn = binop_init, .make_state_type = NULL, .emit_init = NULL, .emit_proc = binop_emit_proc });
    register_opcode((Opcode){ .name = "Sub_aba", .priv = &sub_aba, .special_indices = 0, .init_fn = binop_init, .make_state_type = NULL, .emit_init = NULL, .emit_proc = binop_emit_proc });
    register_opcode((Opcode){ .name = "Sub_baa", .priv = &sub_baa, .special_indices = 0, .init_fn = binop_init, .make_state_type = NULL, .emit_init = NULL, .emit_proc = binop_emit_proc });
    register_opcode((Opcode){ .name = "Sub_bbb", .priv = &sub_bbb, .special_indices = 0, .init_fn = binop_init, .make_state_type = NULL, .emit_init = NULL, .emit_proc = binop_emit_proc });

    register_opcode((Opcode){ .name = "Div_aaa", .priv = &div_aaa, .special_indices = 0, .init_fn = binop_init, .make_state_type = NULL, .emit_init = NULL, .emit_proc = binop_emit_proc });
    register_opcode((Opcode){ .name = "Div_aba", .priv = &div_aba, .special_indices = 0, .init_fn = binop_init, .make_state_type = NULL, .emit_init = NULL, .emit_proc = binop_emit_proc });
    register_opcode((Opcode){ .name = "Div_baa", .priv = &div_baa, .special_indices = 0, .init_fn = binop_init, .make_state_type = NULL, .emit_init = NULL, .emit_proc = binop_emit_proc });
    register_opcode((Opcode){ .name = "Div_bbb", .priv = &div_bbb, .special_indices = 0, .init_fn = binop_init, .make_state_type = NULL, .emit_init = NULL, .emit_proc = binop_emit_proc });

    register_opcode((Opcode){
        .name = "Const_b",
        .priv = NULL,
        .special_indices = 1,
        .init_fn = NULL,
        .make_state_type = NULL,
        .emit_init = NULL,
        .emit_proc = const_emit_proc
    });
    register_opcode((Opcode){
        .name = "Control_b",
        .priv = NULL,
        .special_indices = 1,
        .init_fn = NULL,
        .make_state_type = NULL,
        .emit_init = NULL,
        .emit_proc = control_emit_proc
    });
}

gcc_jit_result *compile(const struct dag *g, unsigned int sample_rate)
{
    gcc_jit_context *ctx = gcc_jit_context_acquire();
    if (!ctx) { fprintf(stderr, "failed to acquire jit context\n"); exit(1); }

    gcc_jit_context_set_int_option(ctx, GCC_JIT_INT_OPTION_OPTIMIZATION_LEVEL, 3);
    gcc_jit_context_set_bool_option(ctx, GCC_JIT_BOOL_OPTION_DUMP_INITIAL_GIMPLE, 1);
    gcc_jit_context_set_bool_option(ctx, GCC_JIT_BOOL_OPTION_DUMP_GENERATED_CODE, 1);

    const Opcode **ops = calloc(g->nVertices, sizeof(*ops));
    if (!ops) {
        gcc_jit_context_release(ctx);
        return NULL;
    }
    for (size_t i = 0; i < g->nVertices; i++)
        ops[i] = find_opcode(g->vertices, i);

    gcc_jit_lvalue **state_for_vertex = calloc(g->nVertices, sizeof(*state_for_vertex));
    if (!state_for_vertex) {
        gcc_jit_context_release(ctx);
        free(ops);
        return NULL;
    }

    for (size_t i = 0; i < g->nVertices; i++) {
        const Opcode *op = ops[i];

        gcc_jit_type *t_state_i = NULL;
        if (op->make_state_type) {
            t_state_i = op->make_state_type(op, ctx);
            if (!t_state_i) {
                gcc_jit_context_release(ctx);
                free(ops);
                free(state_for_vertex);
                return NULL;
            }

            char gname[32];
            snprintf(gname, sizeof(gname), "s%zu", i);
            state_for_vertex[i] = gcc_jit_context_new_global(ctx, NULL,
                GCC_JIT_GLOBAL_INTERNAL, t_state_i, gname
            );
        }

        if (op->init_fn) op->init_fn(op, ctx, t_state_i, sample_rate);
    }

    gcc_jit_type *t_void = gcc_jit_context_get_type(ctx, GCC_JIT_TYPE_VOID);
    gcc_jit_function *fn_init = gcc_jit_context_new_function(ctx, NULL,
        GCC_JIT_FUNCTION_EXPORTED, t_void, "init", 0, NULL, 0
    );
    gcc_jit_block *init_entry = gcc_jit_function_new_block(fn_init, "entry");

    for (size_t i = 0; i < g->nVertices; i++) {
        const Opcode *op = ops[i];

        gcc_jit_lvalue *lv_state = state_for_vertex[i];
        if (lv_state && op->emit_init)
            op->emit_init(op, ctx, init_entry, lv_state);
    }

    gcc_jit_block_end_with_void_return(init_entry, NULL);

    // -------------------------------------------------------------------------
    // process(const float *controls, const float *in, size_t inC,
    //         float *out, size_t outC)
    // -------------------------------------------------------------------------
    gcc_jit_type *t_size_t = gcc_jit_context_get_type(ctx, GCC_JIT_TYPE_SIZE_T);
    gcc_jit_type *t_float  = gcc_jit_context_get_type(ctx, GCC_JIT_TYPE_FLOAT);
    gcc_jit_type *t_float_ptr = gcc_jit_type_get_pointer(t_float);
    gcc_jit_type *t_const_float_ptr = gcc_jit_type_get_pointer(gcc_jit_type_get_const(t_float));

    gcc_jit_param *p_controls =
        gcc_jit_context_new_param(ctx, NULL, t_const_float_ptr, "controls");
    gcc_jit_param *p_in =
        gcc_jit_context_new_param(ctx, NULL, t_const_float_ptr, "in");
    gcc_jit_param *p_inC =
        gcc_jit_context_new_param(ctx, NULL, t_size_t, "inC");
    gcc_jit_param *p_out =
        gcc_jit_context_new_param(ctx, NULL, t_float_ptr, "out");
    gcc_jit_param *p_outC =
        gcc_jit_context_new_param(ctx, NULL, t_size_t, "outC");

    gcc_jit_param *proc_params[] = { p_controls, p_in, p_inC, p_out, p_outC };

    gcc_jit_function *fn_process =
        gcc_jit_context_new_function(ctx, NULL, GCC_JIT_FUNCTION_EXPORTED,
                                     t_void, "process",
                                     5, proc_params, 0);
    {
        (void)p_in; (void)p_inC; (void)p_outC;

        gcc_jit_block *entry = gcc_jit_function_new_block(fn_process, "entry");
        gcc_jit_block *copy_cond = gcc_jit_function_new_block(fn_process, "copy_cond");
        gcc_jit_block *copy_body = gcc_jit_function_new_block(fn_process, "copy_body");
        gcc_jit_block *copy_inc  = gcc_jit_function_new_block(fn_process, "copy_inc");
        gcc_jit_block *done      = gcc_jit_function_new_block(fn_process, "done");

        gcc_jit_rvalue **values = calloc(g->nVertices, sizeof(*values));
        char *rates = calloc(g->nVertices, sizeof(*rates));
        if (!values || !rates) {
            gcc_jit_context_release(ctx);
            free(values);
            free(rates);
            free(state_for_vertex);
            free(ops);
            return NULL;
        }

        /* Precondition: vertices must be topologically sorted so that every
           vertex only depends on values computed earlier in this loop. */
        for (size_t i = 0; i < g->nVertices; i++) {
            const struct vertex *v = &g->vertices[i];
            const Opcode *op = ops[i];

            struct Arg *a = NULL;
            size_t n_args = v->nArgs;

            if (op->special_indices) {
                a = NULL;
                n_args = 0;
            } else if (n_args) {
                a = (struct Arg *)calloc(n_args, sizeof(*a));
                if (!a) {
                    gcc_jit_context_release(ctx);
                    free(values);
                    free(rates);
                    free(state_for_vertex);
                    free(ops);
                    return NULL;
                }
                for (size_t k = 0; k < n_args; k++) {
                    size_t src = v->args[k];
                    a[k].rv = values[src];
                    a[k].rate = rates[src];
                }
            }

            gcc_jit_lvalue *lv_state_field = state_for_vertex[i];

            gcc_jit_rvalue *rv =
                op->emit_proc(op, g, i, ctx, fn_process, entry,
                              lv_state_field,
                              gcc_jit_param_as_rvalue(p_controls),
                              a, n_args,
                              v->rate);

            values[i] = rv;
            rates[i] = v->rate;

            free(a);
        }

        gcc_jit_rvalue *c_0_size  = gcc_jit_context_new_rvalue_from_long(ctx, t_size_t, 0);
        gcc_jit_rvalue *c_BS_size = gcc_jit_context_new_rvalue_from_long(ctx, t_size_t, (long)BS_V);

        assert(g->nVertices > 0);
        size_t out_v = g->nVertices - 1;
        assert(rates[out_v] == 'a');

        gcc_jit_lvalue *lv_i = gcc_jit_function_new_local(fn_process, NULL, t_size_t, "i");
        gcc_jit_block_add_assignment(entry, NULL, lv_i, c_0_size);
        gcc_jit_block_end_with_jump(entry, NULL, copy_cond);

        gcc_jit_rvalue *cnd =
            gcc_jit_context_new_comparison(ctx, NULL, GCC_JIT_COMPARISON_LT,
                                           gcc_jit_lvalue_as_rvalue(lv_i), c_BS_size);
        gcc_jit_block_end_with_conditional(copy_cond, NULL, cnd, copy_body, done);

        gcc_jit_lvalue *lv_out_i =
            gcc_jit_context_new_array_access(ctx, NULL,
                                             gcc_jit_param_as_rvalue(p_out),
                                             gcc_jit_lvalue_as_rvalue(lv_i));
        gcc_jit_lvalue *lv_src_i =
            gcc_jit_context_new_array_access(ctx, NULL,
                                             values[out_v],
                                             gcc_jit_lvalue_as_rvalue(lv_i));
        gcc_jit_block_add_assignment(copy_body, NULL, lv_out_i, gcc_jit_lvalue_as_rvalue(lv_src_i));
        gcc_jit_block_end_with_jump(copy_body, NULL, copy_inc);

        gcc_jit_rvalue *i_next =
            gcc_jit_context_new_binary_op(ctx, NULL, GCC_JIT_BINARY_OP_PLUS, t_size_t,
                                          gcc_jit_lvalue_as_rvalue(lv_i),
                                          gcc_jit_context_new_rvalue_from_long(ctx, t_size_t, 1));
        gcc_jit_block_add_assignment(copy_inc, NULL, lv_i, i_next);
        gcc_jit_block_end_with_jump(copy_inc, NULL, copy_cond);

        gcc_jit_block_end_with_void_return(done, NULL);

        free(values);
        free(rates);
    }

    free(state_for_vertex);
    free(ops);

    gcc_jit_result *res = gcc_jit_context_compile(ctx);
    if (!res)
        die(ctx, "JIT compile failed");

    gcc_jit_context_release(ctx);
    return res;
}

int main() {
    register_builtin_opcodes();
    gcc_jit_result *r = compile(&graph, 44100u);

    void (*init)() = gcc_jit_result_get_code(r, "init");
    void (*process)(const float*, const float*, size_t, float*, size_t) =
        gcc_jit_result_get_code(r, "process");

    init();

    float controls[1] = { 440.0f };
    for (size_t n = 0; n < 10; n++) {
        float out[BS_V];
        process(controls, NULL, 0, out, 1);
        for (size_t i = 0; i < BS_V; i++) {
            printf("%f\n", out[i]);
        }
    }

    gcc_jit_result_release(r);
}
