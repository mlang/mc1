#include <libgccjit.h>
#include <assert.h>
#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const float PI_F = 3.14f;
static const float SAMPLE_RATE_F = 44100.0f;
static const size_t BS_V = 128;

enum { FREQ };

static void die(gcc_jit_context *ctx, const char *msg)
{
    fprintf(stderr, "%s\n", msg);
    fprintf(stderr, "%s\n", gcc_jit_context_get_first_error(ctx));
    exit(1);
}

gcc_jit_rvalue *decay_array_to_pointer(gcc_jit_context *ctx, gcc_jit_lvalue *lvalue)
{
    gcc_jit_type *t_size_t = gcc_jit_context_get_type(ctx, GCC_JIT_TYPE_SIZE_T);
    return gcc_jit_lvalue_get_address(gcc_jit_context_new_array_access(ctx, NULL, gcc_jit_lvalue_as_rvalue(lvalue), gcc_jit_context_new_rvalue_from_long(ctx, t_size_t, 0)), NULL);
}

gcc_jit_result *build_sine_object(void)
{
    gcc_jit_context *ctx = gcc_jit_context_acquire();
    if (!ctx) { fprintf(stderr, "failed to acquire jit context\n"); exit(1); }

    gcc_jit_context_set_int_option(ctx, GCC_JIT_INT_OPTION_OPTIMIZATION_LEVEL, 2);

    // Types
    gcc_jit_type *t_void   = gcc_jit_context_get_type(ctx, GCC_JIT_TYPE_VOID);
    gcc_jit_type *t_size_t = gcc_jit_context_get_type(ctx, GCC_JIT_TYPE_SIZE_T);
    gcc_jit_type *t_float  = gcc_jit_context_get_type(ctx, GCC_JIT_TYPE_FLOAT);

    gcc_jit_type *t_float_ptr = gcc_jit_type_get_pointer(t_float);
    gcc_jit_type *t_const_float_ptr = gcc_jit_type_get_pointer(gcc_jit_type_get_const(t_float));

    // Constants
    gcc_jit_rvalue *c_PI          = gcc_jit_context_new_rvalue_from_double(ctx, t_float, PI_F);
    gcc_jit_rvalue *c_SR          = gcc_jit_context_new_rvalue_from_double(ctx, t_float, SAMPLE_RATE_F);
    gcc_jit_rvalue *c_0_size     = gcc_jit_context_new_rvalue_from_long(ctx, t_size_t, 0);
    gcc_jit_rvalue *c_BS_size     = gcc_jit_context_new_rvalue_from_long(ctx, t_size_t, (long)BS_V);
    gcc_jit_rvalue *c_0_f         = gcc_jit_context_new_rvalue_from_double(ctx, t_float, 0.0);
    gcc_jit_rvalue *c_02_f        = gcc_jit_context_new_rvalue_from_double(ctx, t_float, 0.2);
    gcc_jit_rvalue *c_2_f         = gcc_jit_context_new_rvalue_from_double(ctx, t_float, 2.0);

    // External: float sinf(float)
    gcc_jit_param *p_sinf_x = gcc_jit_context_new_param(ctx, NULL, t_float, "x");
    gcc_jit_function *fn_sinf =
        gcc_jit_context_new_function(ctx, NULL,
                                     GCC_JIT_FUNCTION_IMPORTED,
                                     t_float, "sinf",
                                     1, &p_sinf_x, 0);

    // struct sinosc_state { float phase; };
    gcc_jit_field *fld_phase = gcc_jit_context_new_field(ctx, NULL, t_float, "phase");
    gcc_jit_struct *st_sinosc = gcc_jit_context_new_struct_type(ctx, NULL, "sinosc_state", 1, &fld_phase);
    gcc_jit_type *t_sinosc = gcc_jit_struct_as_type(st_sinosc);
    gcc_jit_type *t_sinosc_ptr = gcc_jit_type_get_pointer(t_sinosc);

    // struct state { struct sinosc_state e2; };
    gcc_jit_field *fld_e2 = gcc_jit_context_new_field(ctx, NULL, t_sinosc, "e2");
    gcc_jit_struct *st_state = gcc_jit_context_new_struct_type(ctx, NULL, "state", 1, &fld_e2);
    gcc_jit_type *t_state = gcc_jit_struct_as_type(st_state);

    // static struct state s;
    gcc_jit_lvalue *gv_s =
        gcc_jit_context_new_global(ctx, NULL, GCC_JIT_GLOBAL_INTERNAL, t_state, "s");

    // -------------------------------------------------------------------------
    // init_sinosc(struct sinosc_state *state) { state->phase = 0; }
    // -------------------------------------------------------------------------
    gcc_jit_param *p_init_state =
        gcc_jit_context_new_param(ctx, NULL, t_sinosc_ptr, "state");
    gcc_jit_function *fn_init_sinosc =
        gcc_jit_context_new_function(ctx, NULL, GCC_JIT_FUNCTION_INTERNAL,
                                     t_void, "init_sinosc",
                                     1, &p_init_state, 0);
    {
        gcc_jit_block *b = gcc_jit_function_new_block(fn_init_sinosc, "entry");
        gcc_jit_lvalue *lv_phase =
            gcc_jit_lvalue_access_field(
                gcc_jit_rvalue_dereference(gcc_jit_param_as_rvalue(p_init_state), NULL),
                NULL, fld_phase);

        gcc_jit_block_add_assignment(b, NULL, lv_phase, c_0_f);
        gcc_jit_block_end_with_void_return(b, NULL);
    }

    // -------------------------------------------------------------------------
    // process_sinosc_bb(state, output, frequency, phase_offset)
    // -------------------------------------------------------------------------
    gcc_jit_param *p_ps_state =
        gcc_jit_context_new_param(ctx, NULL, t_sinosc_ptr, "state");
    gcc_jit_param *p_ps_output =
        gcc_jit_context_new_param(ctx, NULL, t_float_ptr, "output");
    gcc_jit_param *p_ps_freq =
        gcc_jit_context_new_param(ctx, NULL, t_float, "frequency");
    gcc_jit_param *p_ps_off =
        gcc_jit_context_new_param(ctx, NULL, t_float, "phase_offset");

    gcc_jit_param *ps_params[] = { p_ps_state, p_ps_output, p_ps_freq, p_ps_off };

    gcc_jit_function *fn_process_sinosc_bb =
        gcc_jit_context_new_function(ctx, NULL, GCC_JIT_FUNCTION_INTERNAL,
                                     t_void, "process_sinosc_bb",
                                     4, ps_params, 0);
    {
        gcc_jit_block *entry = gcc_jit_function_new_block(fn_process_sinosc_bb, "entry");
        gcc_jit_block *loop_cond = gcc_jit_function_new_block(fn_process_sinosc_bb, "loop_cond");
        gcc_jit_block *loop_body = gcc_jit_function_new_block(fn_process_sinosc_bb, "loop_body");
        gcc_jit_block *wrap_then = gcc_jit_function_new_block(fn_process_sinosc_bb, "wrap_then");
        gcc_jit_block *wrap_join = gcc_jit_function_new_block(fn_process_sinosc_bb, "wrap_join");
        gcc_jit_block *loop_inc  = gcc_jit_function_new_block(fn_process_sinosc_bb, "loop_inc");
        gcc_jit_block *done      = gcc_jit_function_new_block(fn_process_sinosc_bb, "done");

        // locals: float phase; float phase_increment; size_t i;
        gcc_jit_lvalue *lv_phase =
            gcc_jit_function_new_local(fn_process_sinosc_bb, NULL, t_float, "phase");
        gcc_jit_lvalue *lv_inc =
            gcc_jit_function_new_local(fn_process_sinosc_bb, NULL, t_float, "phase_increment");
        gcc_jit_lvalue *lv_i =
            gcc_jit_function_new_local(fn_process_sinosc_bb, NULL, t_size_t, "i");

        // phase = state->phase
        gcc_jit_lvalue *lv_state_phase =
            gcc_jit_lvalue_access_field(
                gcc_jit_rvalue_dereference(gcc_jit_param_as_rvalue(p_ps_state), NULL),
                NULL, fld_phase);

        gcc_jit_block_add_assignment(entry, NULL, lv_phase, gcc_jit_lvalue_as_rvalue(lv_state_phase));

        // phase_increment = 2*PI*frequency/SAMPLE_RATE
        gcc_jit_rvalue *two_pi =
            gcc_jit_context_new_binary_op(ctx, NULL, GCC_JIT_BINARY_OP_MULT, t_float,
                                          c_2_f, c_PI);
        gcc_jit_rvalue *num =
            gcc_jit_context_new_binary_op(ctx, NULL, GCC_JIT_BINARY_OP_MULT, t_float,
                                          two_pi, gcc_jit_param_as_rvalue(p_ps_freq));
        gcc_jit_rvalue *inc =
            gcc_jit_context_new_binary_op(ctx, NULL, GCC_JIT_BINARY_OP_DIVIDE, t_float,
                                          num, c_SR);
        gcc_jit_block_add_assignment(entry, NULL, lv_inc, inc);

        // i = 0; goto loop_cond
        gcc_jit_block_add_assignment(entry, NULL, lv_i, c_0_size);
        gcc_jit_block_end_with_jump(entry, NULL, loop_cond);

        // loop_cond: i < BS ?
        gcc_jit_rvalue *cond =
            gcc_jit_context_new_comparison(ctx, NULL, GCC_JIT_COMPARISON_LT,
                                           gcc_jit_lvalue_as_rvalue(lv_i),
                                           c_BS_size);
        gcc_jit_block_end_with_conditional(loop_cond, NULL, cond, loop_body, done);

        // loop_body:
        // output[i] = sinf(phase + phase_offset);
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

        // phase += phase_increment
        gcc_jit_rvalue *phase_next =
            gcc_jit_context_new_binary_op(ctx, NULL, GCC_JIT_BINARY_OP_PLUS, t_float,
                                          gcc_jit_lvalue_as_rvalue(lv_phase),
                                          gcc_jit_lvalue_as_rvalue(lv_inc));
        gcc_jit_block_add_assignment(loop_body, NULL, lv_phase, phase_next);

        // if (phase > 2*PI) phase -= 2*PI
        gcc_jit_rvalue *two_pi_again = two_pi; // reuse
        gcc_jit_rvalue *wrap_cond =
            gcc_jit_context_new_comparison(ctx, NULL, GCC_JIT_COMPARISON_GT,
                                           gcc_jit_lvalue_as_rvalue(lv_phase),
                                           two_pi_again);
        gcc_jit_block_end_with_conditional(loop_body, NULL, wrap_cond, wrap_then, wrap_join);

        gcc_jit_rvalue *phase_wrapped =
            gcc_jit_context_new_binary_op(ctx, NULL, GCC_JIT_BINARY_OP_MINUS, t_float,
                                          gcc_jit_lvalue_as_rvalue(lv_phase),
                                          two_pi_again);
        gcc_jit_block_add_assignment(wrap_then, NULL, lv_phase, phase_wrapped);
        gcc_jit_block_end_with_jump(wrap_then, NULL, wrap_join);

        // wrap_join -> loop_inc
        gcc_jit_block_end_with_jump(wrap_join, NULL, loop_inc);

        // i++
        gcc_jit_rvalue *i_next =
            gcc_jit_context_new_binary_op(ctx, NULL, GCC_JIT_BINARY_OP_PLUS, t_size_t,
                                          gcc_jit_lvalue_as_rvalue(lv_i),
                                          gcc_jit_context_new_rvalue_from_long(ctx, t_size_t, 1));
        gcc_jit_block_add_assignment(loop_inc, NULL, lv_i, i_next);
        gcc_jit_block_end_with_jump(loop_inc, NULL, loop_cond);

        // done: state->phase = phase; return;
        gcc_jit_block_add_assignment(done, NULL, lv_state_phase, gcc_jit_lvalue_as_rvalue(lv_phase));
        gcc_jit_block_end_with_void_return(done, NULL);
    }

    // -------------------------------------------------------------------------
    // process_mul_ab(r, a, b)
    // -------------------------------------------------------------------------
    gcc_jit_param *p_pm_r =
        gcc_jit_context_new_param(ctx, NULL, t_float_ptr, "r");
    gcc_jit_param *p_pm_a =
        gcc_jit_context_new_param(ctx, NULL, t_const_float_ptr, "a");
    gcc_jit_param *p_pm_b =
        gcc_jit_context_new_param(ctx, NULL, t_float, "b");
    gcc_jit_param *pm_params[] = { p_pm_r, p_pm_a, p_pm_b };

    gcc_jit_function *fn_process_mul_ab =
        gcc_jit_context_new_function(ctx, NULL, GCC_JIT_FUNCTION_INTERNAL,
                                     t_void, "process_mul_ab",
                                     3, pm_params, 0);
    {
        gcc_jit_block *entry = gcc_jit_function_new_block(fn_process_mul_ab, "entry");
        gcc_jit_block *cond  = gcc_jit_function_new_block(fn_process_mul_ab, "cond");
        gcc_jit_block *body  = gcc_jit_function_new_block(fn_process_mul_ab, "body");
        gcc_jit_block *inc   = gcc_jit_function_new_block(fn_process_mul_ab, "inc");
        gcc_jit_block *done  = gcc_jit_function_new_block(fn_process_mul_ab, "done");

        gcc_jit_lvalue *lv_i = gcc_jit_function_new_local(fn_process_mul_ab, NULL, t_size_t, "i");
        gcc_jit_block_add_assignment(entry, NULL, lv_i, c_0_size);
        gcc_jit_block_end_with_jump(entry, NULL, cond);

        gcc_jit_rvalue *cnd =
            gcc_jit_context_new_comparison(ctx, NULL, GCC_JIT_COMPARISON_LT,
                                           gcc_jit_lvalue_as_rvalue(lv_i), c_BS_size);
        gcc_jit_block_end_with_conditional(cond, NULL, cnd, body, done);

        gcc_jit_lvalue *lv_r_i =
            gcc_jit_context_new_array_access(ctx, NULL,
                                             gcc_jit_param_as_rvalue(p_pm_r),
                                             gcc_jit_lvalue_as_rvalue(lv_i));
        gcc_jit_lvalue *lv_a_i =
            gcc_jit_context_new_array_access(ctx, NULL,
                                             gcc_jit_param_as_rvalue(p_pm_a),
                                             gcc_jit_lvalue_as_rvalue(lv_i));
        gcc_jit_rvalue *mul =
            gcc_jit_context_new_binary_op(ctx, NULL, GCC_JIT_BINARY_OP_MULT, t_float,
                                          gcc_jit_lvalue_as_rvalue(lv_a_i), gcc_jit_param_as_rvalue(p_pm_b));
        gcc_jit_block_add_assignment(body, NULL, lv_r_i, mul);
        gcc_jit_block_end_with_jump(body, NULL, inc);

        gcc_jit_rvalue *i_next =
            gcc_jit_context_new_binary_op(ctx, NULL, GCC_JIT_BINARY_OP_PLUS, t_size_t,
                                          gcc_jit_lvalue_as_rvalue(lv_i),
                                          gcc_jit_context_new_rvalue_from_long(ctx, t_size_t, 1));
        gcc_jit_block_add_assignment(inc, NULL, lv_i, i_next);
        gcc_jit_block_end_with_jump(inc, NULL, cond);

        gcc_jit_block_end_with_void_return(done, NULL);
    }

    // -------------------------------------------------------------------------
    // void init() { init_sinosc(&s.e2); }
    // -------------------------------------------------------------------------
    gcc_jit_function *fn_init =
        gcc_jit_context_new_function(ctx, NULL, GCC_JIT_FUNCTION_EXPORTED,
                                     t_void, "init", 0, NULL, 0);
    {
        gcc_jit_block *b = gcc_jit_function_new_block(fn_init, "entry");

        gcc_jit_lvalue *lv_s_e2 = gcc_jit_lvalue_access_field(gv_s, NULL, fld_e2);
        gcc_jit_rvalue *addr_s_e2 = gcc_jit_lvalue_get_address(lv_s_e2, NULL);

        gcc_jit_rvalue *args[] = { addr_s_e2 };
        gcc_jit_block_add_eval(b, NULL, gcc_jit_context_new_call(ctx, NULL, fn_init_sinosc, 1, args));
        gcc_jit_block_end_with_void_return(b, NULL);
    }

    // -------------------------------------------------------------------------
    // void process(const float *controls, const float *in, size_t inC,
    //              float *out, size_t outC)
    // -------------------------------------------------------------------------
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

        // e0 = controls[FREQ]
        gcc_jit_rvalue *idx0 = gcc_jit_context_new_rvalue_from_long(ctx, t_size_t, (long)FREQ);
        gcc_jit_lvalue *e0 =
            gcc_jit_context_new_array_access(ctx, NULL,
                                             gcc_jit_param_as_rvalue(p_controls),
                                             idx0);
        gcc_jit_rvalue *e1 = c_0_f;

        // locals: float e2[BS], e4[BS]
        gcc_jit_type *t_float_array_BS = gcc_jit_context_new_array_type(ctx, NULL, t_float, (int)BS_V);
        gcc_jit_lvalue *lv_e2 = gcc_jit_function_new_local(fn_process, NULL, t_float_array_BS, "e2");
        gcc_jit_lvalue *lv_e4 = gcc_jit_function_new_local(fn_process, NULL, t_float_array_BS, "e4");

        // &s.e2
        gcc_jit_lvalue *lv_s_e2 = gcc_jit_lvalue_access_field(gv_s, NULL, fld_e2);
        gcc_jit_rvalue *addr_s_e2 = gcc_jit_lvalue_get_address(lv_s_e2, NULL);

        // call process_sinosc_bb(&s.e2, e2, e0, e1)
        gcc_jit_rvalue *args1[] = { addr_s_e2, decay_array_to_pointer(ctx, lv_e2), gcc_jit_lvalue_as_rvalue(e0), e1 };
        gcc_jit_block_add_eval(entry, NULL,
            gcc_jit_context_new_call(ctx, NULL, fn_process_sinosc_bb, 4, args1));

        // call process_mul_ab(e4, e2, 0.2f)
        gcc_jit_rvalue *args2[] = { decay_array_to_pointer(ctx, lv_e4), decay_array_to_pointer(ctx, lv_e2), c_02_f };
        gcc_jit_block_add_eval(entry, NULL,
            gcc_jit_context_new_call(ctx, NULL, fn_process_mul_ab, 3, args2));

        // for i: out[i] = e4[i]
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
        gcc_jit_lvalue *lv_e4_i =
            gcc_jit_context_new_array_access(ctx, NULL,
                                             gcc_jit_lvalue_as_rvalue(lv_e4),
                                             gcc_jit_lvalue_as_rvalue(lv_i));
        gcc_jit_block_add_assignment(copy_body, NULL, lv_out_i, gcc_jit_lvalue_as_rvalue(lv_e4_i));
        gcc_jit_block_end_with_jump(copy_body, NULL, copy_inc);

        gcc_jit_rvalue *i_next =
            gcc_jit_context_new_binary_op(ctx, NULL, GCC_JIT_BINARY_OP_PLUS, t_size_t,
                                          gcc_jit_lvalue_as_rvalue(lv_i),
                                          gcc_jit_context_new_rvalue_from_long(ctx, t_size_t, 1));
        gcc_jit_block_add_assignment(copy_inc, NULL, lv_i, i_next);
        gcc_jit_block_end_with_jump(copy_inc, NULL, copy_cond);

        gcc_jit_block_end_with_void_return(done, NULL);
    }

    gcc_jit_result *res = gcc_jit_context_compile(ctx);
    if (!res)
        die(ctx, "JIT compile failed");

    gcc_jit_context_release(ctx);
    return res;
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

const float CONSTS[] = { 0.0f, 0.2f };

const size_t V0_ARGS[] = { 0 };
const size_t V1_ARGS[] = { 0 };
const size_t V2_ARGS[] = { 0, 1 };
const size_t V3_ARGS[] = { 1 };
const size_t V4_ARGS[] = { 2, 3 };

const struct vertex VERTICES[] = {
  { "Control", 'b', 1, V0_ARGS },
  { "Const", 'b', 1, V1_ARGS },
  { "SinOsc", 'a', 2, V2_ARGS },
  { "Const", 'b', 1, V3_ARGS },
  { "Mul", 'a', 2, V4_ARGS }
};

struct dag graph = {
  .constants = CONSTS,
  .vertices = VERTICES,
  .nVertices = 5
};

/* --- Opcode registry --- */

typedef struct Opcode Opcode;

typedef void (*opcode_init_fn)(const Opcode *op, gcc_jit_context *ctx, gcc_jit_type *t_state);
typedef gcc_jit_type *(*make_state_type_fn)(const Opcode *op, gcc_jit_context *ctx);
typedef void (*emit_init_fn)(const Opcode *op, gcc_jit_context *ctx, gcc_jit_block *entry, gcc_jit_lvalue *lv_state_field);

struct Opcode {
    const char *name;
    void *priv;                         /* private per-opcode data */
    opcode_init_fn init_fn;             /* optional (may be NULL); called before anything else */
    make_state_type_fn make_state_type; /* optional (may be NULL) */
    emit_init_fn emit_init;             /* optional (may be NULL) */
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
    const Opcode *res =
        (const Opcode *)bsearch(&key, g_opcodes, g_n_opcodes, sizeof(Opcode), opcode_cmp_by_name);

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
    Opcode *new_ops = (Opcode *)realloc(g_opcodes, (g_n_opcodes + 1) * sizeof(*new_ops));
    if (!new_ops) return 0;
    g_opcodes = new_ops;
    g_opcodes[g_n_opcodes++] = op;
    qsort(g_opcodes, g_n_opcodes, sizeof(Opcode), opcode_cmp_by_name);
    return 1;
}

/* SinOsc: has per-instance state { float phase; } */
struct sinosc_priv {
    gcc_jit_field *fld_phase;
    gcc_jit_struct *st_sinosc;
    gcc_jit_function *proc;
};

static gcc_jit_type *sinosc_make_state_type(const Opcode *op, gcc_jit_context *ctx)
{
    struct sinosc_priv *p = (struct sinosc_priv *)op->priv;

    if (!p->fld_phase) {
        gcc_jit_type *t_float = gcc_jit_context_get_type(ctx, GCC_JIT_TYPE_FLOAT);
        p->fld_phase = gcc_jit_context_new_field(ctx, NULL, t_float, "phase");
    }
    if (!p->st_sinosc) {
        p->st_sinosc = gcc_jit_context_new_struct_type(ctx, NULL, "sinosc_state", 1, &p->fld_phase);
    }
    return gcc_jit_struct_as_type(p->st_sinosc);
}

static void sinosc_emit_init(const Opcode *op, gcc_jit_context *ctx, gcc_jit_block *entry, gcc_jit_lvalue *lv_state_field)
{
    struct sinosc_priv *p = (struct sinosc_priv *)op->priv;
    gcc_jit_type *t_float = gcc_jit_context_get_type(ctx, GCC_JIT_TYPE_FLOAT);

    assert(p->fld_phase);

    gcc_jit_lvalue *lv_phase =
        gcc_jit_lvalue_access_field(lv_state_field, NULL, p->fld_phase);

    gcc_jit_rvalue *c_0_f = gcc_jit_context_new_rvalue_from_double(ctx, t_float, 0.0);
    gcc_jit_block_add_assignment(entry, NULL, lv_phase, c_0_f);
}

static void sinosc_init(const Opcode *op, gcc_jit_context *ctx, gcc_jit_type *t_state)
{
    struct sinosc_priv *p = (struct sinosc_priv *)op->priv;
    if (p->proc)
        return;

    gcc_jit_type *t_void   = gcc_jit_context_get_type(ctx, GCC_JIT_TYPE_VOID);
    gcc_jit_type *t_size_t = gcc_jit_context_get_type(ctx, GCC_JIT_TYPE_SIZE_T);
    gcc_jit_type *t_float  = gcc_jit_context_get_type(ctx, GCC_JIT_TYPE_FLOAT);

    gcc_jit_type *t_float_ptr = gcc_jit_type_get_pointer(t_float);
    gcc_jit_type *t_state_ptr = gcc_jit_type_get_pointer(t_state);

    gcc_jit_rvalue *c_0_size  = gcc_jit_context_new_rvalue_from_long(ctx, t_size_t, 0);
    gcc_jit_rvalue *c_BS_size = gcc_jit_context_new_rvalue_from_long(ctx, t_size_t, (long)BS_V);

    assert(p->fld_phase);

    /* External: float sinf(float) */
    gcc_jit_param *p_sinf_x = gcc_jit_context_new_param(ctx, NULL, t_float, "x");
    gcc_jit_function *fn_sinf =
        gcc_jit_context_new_function(ctx, NULL,
                                     GCC_JIT_FUNCTION_IMPORTED,
                                     t_float, "sinf",
                                     1, &p_sinf_x, 0);

    /* void sinosc_process_bba(state*, out*, freq, phase_offset) */
    gcc_jit_param *p_ps_state =
        gcc_jit_context_new_param(ctx, NULL, t_state_ptr, "state");
    gcc_jit_param *p_ps_output =
        gcc_jit_context_new_param(ctx, NULL, t_float_ptr, "output");
    gcc_jit_param *p_ps_freq =
        gcc_jit_context_new_param(ctx, NULL, t_float, "frequency");
    gcc_jit_param *p_ps_off =
        gcc_jit_context_new_param(ctx, NULL, t_float, "phase_offset");

    gcc_jit_param *ps_params[] = { p_ps_state, p_ps_output, p_ps_freq, p_ps_off };

    p->proc =
        gcc_jit_context_new_function(ctx, NULL, GCC_JIT_FUNCTION_INTERNAL,
                                     t_void, "sinosc_process_bba",
                                     4, ps_params, 0);

    gcc_jit_block *entry     = gcc_jit_function_new_block(p->proc, "entry");
    gcc_jit_block *loop_cond = gcc_jit_function_new_block(p->proc, "loop_cond");
    gcc_jit_block *loop_body = gcc_jit_function_new_block(p->proc, "loop_body");
    gcc_jit_block *loop_inc  = gcc_jit_function_new_block(p->proc, "loop_inc");
    gcc_jit_block *done      = gcc_jit_function_new_block(p->proc, "done");

    gcc_jit_lvalue *lv_phase =
        gcc_jit_function_new_local(p->proc, NULL, t_float, "phase");
    gcc_jit_lvalue *lv_i =
        gcc_jit_function_new_local(p->proc, NULL, t_size_t, "i");

    gcc_jit_lvalue *lv_state_phase =
        gcc_jit_lvalue_access_field(
            gcc_jit_rvalue_dereference(gcc_jit_param_as_rvalue(p_ps_state), NULL),
            NULL, p->fld_phase);

    gcc_jit_block_add_assignment(entry, NULL, lv_phase, gcc_jit_lvalue_as_rvalue(lv_state_phase));
    gcc_jit_block_add_assignment(entry, NULL, lv_i, c_0_size);
    gcc_jit_block_end_with_jump(entry, NULL, loop_cond);

    gcc_jit_rvalue *cond =
        gcc_jit_context_new_comparison(ctx, NULL, GCC_JIT_COMPARISON_LT,
                                       gcc_jit_lvalue_as_rvalue(lv_i),
                                       c_BS_size);
    gcc_jit_block_end_with_conditional(loop_cond, NULL, cond, loop_body, done);

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
    gcc_jit_block_end_with_jump(loop_body, NULL, loop_inc);

    gcc_jit_rvalue *i_next =
        gcc_jit_context_new_binary_op(ctx, NULL, GCC_JIT_BINARY_OP_PLUS, t_size_t,
                                      gcc_jit_lvalue_as_rvalue(lv_i),
                                      gcc_jit_context_new_rvalue_from_long(ctx, t_size_t, 1));
    gcc_jit_block_add_assignment(loop_inc, NULL, lv_i, i_next);
    gcc_jit_block_end_with_jump(loop_inc, NULL, loop_cond);

    gcc_jit_block_add_assignment(done, NULL, lv_state_phase, gcc_jit_lvalue_as_rvalue(lv_phase));
    gcc_jit_block_end_with_void_return(done, NULL);

    (void)p_ps_freq;
}

/* Call once before build_module */
static void register_builtin_opcodes(void)
{
    static struct sinosc_priv sinosc_p = {0};

    register_opcode((Opcode){
        .name = "SinOsc_bba",
        .priv = &sinosc_p,
        .init_fn = sinosc_init,
        .make_state_type = sinosc_make_state_type,
        .emit_init = sinosc_emit_init
    });
    register_opcode((Opcode){
        .name = "Mul_aba",
        .priv = NULL,
        .init_fn = NULL,
        .make_state_type = NULL,
        .emit_init = NULL
    });
    register_opcode((Opcode){
        .name = "Const_b",
        .priv = NULL,
        .init_fn = NULL,
        .make_state_type = NULL,
        .emit_init = NULL
    });
    register_opcode((Opcode){
        .name = "Control_b",
        .priv = NULL,
        .init_fn = NULL,
        .make_state_type = NULL,
        .emit_init = NULL
    });
}

gcc_jit_result *build_module(const struct dag *g)
{
    gcc_jit_context *ctx = gcc_jit_context_acquire();
    if (!ctx) { fprintf(stderr, "failed to acquire jit context\n"); exit(1); }

    gcc_jit_context_set_bool_option(ctx, GCC_JIT_BOOL_OPTION_DUMP_INITIAL_GIMPLE, 1);
    gcc_jit_context_set_bool_option(ctx, GCC_JIT_BOOL_OPTION_DUMP_GENERATED_CODE, 1);

    const Opcode **ops = (const Opcode **)calloc(g->nVertices, sizeof(*ops));
    if (!ops) {
        gcc_jit_context_release(ctx);
        return NULL;
    }
    for (size_t i = 0; i < g->nVertices; i++)
        ops[i] = find_opcode(g->vertices, i);

    gcc_jit_field **fields = NULL;
    size_t n_fields = 0;
    gcc_jit_field **field_for_vertex = (gcc_jit_field **)calloc(g->nVertices, sizeof(*field_for_vertex));
    if (!field_for_vertex) {
        gcc_jit_context_release(ctx);
        free(ops);
        return NULL;
    }

    for (size_t i = 0; i < g->nVertices; i++) {
        char *mn = mangle(g->vertices, i);
        if (mn) {
            printf("%s\n", mn);
            free(mn);
        }

        const struct vertex *v = &g->vertices[i];
        const Opcode *op = ops[i];

        gcc_jit_type *t_state_i = NULL;
        if (op->make_state_type) {
            t_state_i = op->make_state_type(op, ctx);
            if (!t_state_i) {
                gcc_jit_context_release(ctx);
                free(ops);
                free(fields);
                free(field_for_vertex);
                return NULL;
            }

            char fname[32];
            snprintf(fname, sizeof(fname), "e%zu", i);

            gcc_jit_field *f = gcc_jit_context_new_field(ctx, NULL, t_state_i, fname);

            gcc_jit_field **new_fields = realloc(fields, (n_fields + 1) * sizeof(*new_fields));
            if (!new_fields) {
                gcc_jit_context_release(ctx);
                free(ops);
                free(fields);
                free(field_for_vertex);
                return NULL;
            }
            fields = new_fields;
            fields[n_fields++] = f;
            field_for_vertex[i] = f;
        }

        if (op->init_fn)
            op->init_fn(op, ctx, t_state_i);
    }

    gcc_jit_struct *st_state =
        gcc_jit_context_new_struct_type(ctx, NULL, "state", (int)n_fields, fields);
    gcc_jit_type *t_state = gcc_jit_struct_as_type(st_state);

    gcc_jit_lvalue *gv_s = gcc_jit_context_new_global(ctx, NULL, GCC_JIT_GLOBAL_INTERNAL, t_state, "s");

    gcc_jit_type *t_void = gcc_jit_context_get_type(ctx, GCC_JIT_TYPE_VOID);
    gcc_jit_function *fn_init =
        gcc_jit_context_new_function(ctx, NULL, GCC_JIT_FUNCTION_EXPORTED,
                                     t_void, "init", 0, NULL, 0);
    gcc_jit_block *init_entry = gcc_jit_function_new_block(fn_init, "entry");

    for (size_t i = 0; i < g->nVertices; i++) {
        const struct vertex *v = &g->vertices[i];
        const Opcode *op = ops[i];

        gcc_jit_field *f = field_for_vertex[i];
        if (f && op->emit_init) {
            gcc_jit_lvalue *lv_s_field = gcc_jit_lvalue_access_field(gv_s, NULL, f);
            op->emit_init(op, ctx, init_entry, lv_s_field);
        }
    }

    gcc_jit_block_end_with_void_return(init_entry, NULL);

    free(fields);
    free(field_for_vertex);
    free(ops);

    gcc_jit_result *res = gcc_jit_context_compile(ctx);
    if (!res)
        die(ctx, "JIT compile failed");

    gcc_jit_context_release(ctx);
    return res;
}

int main() {
   gcc_jit_result *r = build_sine_object();
   void (*init)(void) = gcc_jit_result_get_code(r, "init");
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

   /* --- */
   register_builtin_opcodes();
   r = build_module(&graph);
   init = gcc_jit_result_get_code(r, "init");
   init();
   gcc_jit_result_release(r);
}
